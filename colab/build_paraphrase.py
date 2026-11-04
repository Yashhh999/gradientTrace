# gradientTrace - paraphrase attack slice for the frozen gate (paste this whole file into one cell)
# Kaggle: Accelerator > GPU T4 x2, Internet > On. Colab: Runtime > Change runtime type > T4 GPU.
# Takes the frozen gate's AI rows, has a second (never-trained-on) model paraphrase each one, and
# pushes the result as a new JSONL next to the frozen dataset. Scoring it is a later step: it is
# the adversarial half of the gate, so a detector's paraphrase robustness is measured with the same
# headline metric (TPR at a human-calibrated 1% FPR), never by generation fluency.
#
# Paraphrase it as an *edit pass*, not a completion: the model only rewrites, it never composes,
# so the output keeps the human's facts and length while the attack tooling (RAID's synonym /
# paraphrase attacks) is what we are simulating. The second model must be one the detector has
# never seen, exactly like the gate's own generator rule.

import os

PARA_MODEL = "Qwen/Qwen2.5-3B-Instruct"  # second generator; never used in training
GATE_REPO = "Yashhh999/gradientTrace"    # where the frozen gate dataset lives
GATE_NAME = "external_gate_v1"
OUT_NAME = "external_gate_v1_paraphrase.jsonl"
REGENERATE = False   # False restores the pushed slice and adds only the missing paraphrases
BATCH = 8
MAX_NEW = 256
PUSH = True
TEMPERATURE = 0.8
TOP_P = 0.9

import json
import logging
import time

T0 = time.time()


def stage(message):
    print(f"\n[para +{time.time() - T0:5.0f}s] {message}", flush=True)


token = os.environ.get("HF_TOKEN") or ""
if not token:
    try:
        from kaggle_secrets import UserSecretsClient

        token = UserSecretsClient().get_secret("HF_TOKEN") or ""
    except Exception as exc:
        print("[para] Kaggle secret unavailable:", exc, flush=True)
if not token:
    try:
        from google.colab import userdata

        token = userdata.get("HF_TOKEN") or ""
    except Exception as exc:
        print("[para] Colab secret unavailable:", exc, flush=True)
if not token and PUSH:
    raise SystemExit(
        "[para] HF_TOKEN missing. Kaggle: Add-ons > Secrets, add HF_TOKEN (write scope) and enable it "
        "for this notebook; Colab: key icon in the left sidebar. Then re-run."
    )
os.environ["HF_TOKEN"] = token
print("[para] HF_TOKEN present (value not printed)" if token else "[para] no HF_TOKEN; public reads only")

import torch  # noqa: E402  (only after the dependency check)

if not torch.cuda.is_available():
    raise SystemExit("[para] no GPU: Runtime > Change runtime type > T4 GPU, then re-run")
import transformers  # noqa: E402

print(
    f"[para] GPU {torch.cuda.get_device_name(0)} | torch {torch.__version__}"
    f" | transformers {transformers.__version__}",
    flush=True,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", force=True)

from pathlib import Path  # noqa: E402

from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

out_dir = Path("runs/gradientTrace")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / OUT_NAME

api = None
if PUSH:
    api = HfApi(token=os.environ["HF_TOKEN"])

if REGENERATE and out_path.is_file():
    out_path.unlink()

existing = {}
if not REGENERATE and out_path.is_file():
    for line in out_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:  # a killed run can leave a half-written last line
            continue
        if isinstance(rec.get("id"), str):
            existing[rec["id"]] = rec
    print(f"[para] restored {len(existing)} paraphrases already pushed", flush=True)

gate_rows = []
for line in Path(hf_hub_download(GATE_REPO, f"{GATE_NAME}.jsonl", repo_type="model")).read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if line:
        gate_rows.append(json.loads(line))

# Only AI rows get rewritten; the human half of the gate stays identical, so the FPR side of any
# comparison is the same population.
todo = [rec for rec in gate_rows if rec["label"] == 1 and rec["id"] not in existing]
print(f"[para] gate AI rows {sum(1 for r in gate_rows if r['label'] == 1)}; to paraphrase {len(todo)}", flush=True)

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

stage(f"loading paraphraser {PARA_MODEL}")
tok = AutoTokenizer.from_pretrained(PARA_MODEL)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(PARA_MODEL, torch_dtype=torch.float16, device_map="auto")
model.eval()

SYSTEM = (
    "You rewrite text. Rewrite the following text in different words, keeping the same facts, "
    "the same length in words, and the same tone. Do not add facts, do not drop facts, do not "
    "write a title, and do not comment on the text. Return only the rewritten text."
)
done = len(existing)

for start in range(0, len(todo), BATCH):
    chunk = todo[start : start + BATCH]
    prompts = [tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": rec["text"]}],
        tokenize=False,
        add_generation_prompt=True,
    ) for rec in chunk]
    enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        gen = model.generate(
            **enc,
            max_new_tokens=MAX_NEW,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
    for rec, out_ids in zip(chunk, gen):
        rewritten = tok.decode(out_ids[enc["input_ids"].shape[1] :], skip_special_tokens=True).strip()
        if len(rewritten) < 20:  # empty or a stray preamble; keep the original rather than publish junk
            rewritten = rec["text"]
        row = dict(rec)
        row.update({
            "id": rec["id"] + "-para",
            "text": rewritten,
            "generator": PARA_MODEL,
            "prompt": "paraphrase",
        })
        existing[row["id"]] = row
        done += 1

    with out_path.open("a", encoding="utf-8") as fh:
        for rec in chunk:
            fh.write(json.dumps(existing[rec["id"] + "-para"], ensure_ascii=False) + "\n")

    if PUSH and api is not None:
        api.upload_file(
            path_or_fileobj=str(out_path),
            path_in_repo=OUT_NAME,
            repo_id=GATE_REPO,
            commit_message="gradientTrace paraphrase attack slice for the frozen gate",
        )

    if done % (BATCH * 5) == 0 or done == len(todo) + len(existing) - len(todo):
        stage(f"paraphrased {done} AI texts; pushed {OUT_NAME}")

print(f"\n[para] done: {done} paraphrased AI rows at {out_path} in {time.time() - T0:.0f}s", flush=True)

# keep GPU 0 clean: a training cell in this same kernel would otherwise OOM at step 0
import gc  # noqa: E402

del model
gc.collect()
torch.cuda.empty_cache()
