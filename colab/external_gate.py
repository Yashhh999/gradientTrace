# gradientTrace - external acceptance gate (paste this whole file into one cell)
# Kaggle: Accelerator > GPU T4 x2, Internet > On. Colab: Runtime > Change runtime type > T4 GPU.
# Generates AI news/reviews with an unseen instruct model, using each human sample as the
# topic and matching its length, then scores humans and AI with the detector under test.
# Headline numbers: FPR/TPR at a human-calibrated 1% threshold, AUROC, pairwise accuracy and
# a length-bucket breakdown. The frozen dataset is pushed once and reused by later models,
# so v2 is scored on exactly the same text.
MODEL_ID = "Yashhh999/gradientTrace"    # detector under test
MODEL_LABEL = "v1"
GEN_MODEL = "Qwen/Qwen2.5-3B-Instruct"  # unseen generator; 7B is an option on Kaggle 2x T4
GATE_REPO = "Yashhh999/gradientTrace"   # where the frozen gate dataset lives
GATE_NAME = "external_gate_v1"
N_PER_GENRE = 400
GEN_BATCH = 8
TEMPERATURE = 0.8
TOP_P = 0.9
REGENERATE = False    # True rebuilds and re-pushes the frozen JSONL
BATCH_SIZE = 32       # detector eval; an OOM halves it
PUSH = True

if MODEL_ID.rstrip("/") == GATE_REPO and MODEL_LABEL != "v1":
    raise SystemExit(
        f"[gate] MODEL_ID is still the gate repo {GATE_REPO} (the v1 detector) while "
        f"MODEL_LABEL={MODEL_LABEL!r}; this would score v1 and file it under the new label. "
        "Set MODEL_ID to the model under test (e.g. Yashhh999/gradientTrace-v2-pilot)."
    )

import importlib.util
import json
import logging
import os
import re
import socket
import subprocess
import sys
import time

T0 = time.time()
REPO_DIR = "/kaggle/working/gradientTrace" if os.path.isdir("/kaggle/working") else "/content/gradientTrace"


def stage(message):
    print(f"\n[gate +{time.time() - T0:5.0f}s] {message}", flush=True)


token = os.environ.get("HF_TOKEN") or ""
if not token:  # Kaggle Add-ons > Secrets, or Colab key icon; never getpass, it blocks the cell
    try:
        from kaggle_secrets import UserSecretsClient

        token = UserSecretsClient().get_secret("HF_TOKEN") or ""
    except Exception as exc:
        print("[gate] Kaggle secret unavailable:", exc, flush=True)
if not token:
    try:
        from google.colab import userdata

        token = userdata.get("HF_TOKEN") or ""
    except Exception as exc:
        print("[gate] Colab secret unavailable:", exc, flush=True)
if not token:
    raise SystemExit(
        "[gate] HF_TOKEN missing. Kaggle: Add-ons > Secrets, add HF_TOKEN (write scope) and "
        "enable it for this notebook; Colab: key icon in the left sidebar. Then re-run."
    )
os.environ["HF_TOKEN"] = token
print("[gate] HF_TOKEN present (value not printed)", flush=True)

try:  # Kaggle silently blocks clone/HF downloads when Internet is off
    socket.create_connection(("huggingface.co", 443), timeout=10).close()
except OSError as exc:
    raise SystemExit(
        "[gate] no internet: Kaggle Settings > Internet > On (or Colab network is down), "
        "then re-run this cell."
    ) from exc

if not os.path.isdir(os.path.join(REPO_DIR, ".git")):
    stage(f"cloning into {REPO_DIR}")
    subprocess.run(
        ["git", "clone", "--depth", "1", "https://github.com/Yashhh999/gradientTrace.git", REPO_DIR],
        check=True,
    )
else:
    stage("updating the existing clone")
    subprocess.run(["git", "-C", REPO_DIR, "pull", "--ff-only", "--quiet"], check=True)
os.chdir(REPO_DIR)
sys.path.insert(0, REPO_DIR)
print(
    "[gate] repo at",
    subprocess.run(
        ["git", "-C", REPO_DIR, "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip(),
    flush=True,
)

missing = [
    package
    for module, package in (
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("datasets", "datasets"),
        ("sentencepiece", "sentencepiece"),
        ("sklearn", "scikit-learn"),
        ("yaml", "pyyaml"),
        ("safetensors", "safetensors"),
        ("tqdm", "tqdm"),
    )
    if importlib.util.find_spec(module) is None
]
if missing:
    stage(f"installing missing packages: {missing}")
    subprocess.run([sys.executable, "-m", "pip", "-q", "install", *missing], check=True)
else:
    print("[gate] dependencies already present; skipping pip", flush=True)

import torch  # noqa: E402  (only after the dependency check)

if not torch.cuda.is_available():
    raise SystemExit("[gate] no GPU: Runtime > Change runtime type > T4 GPU, then re-run")
import transformers  # noqa: E402

print(
    f"[gate] GPU {torch.cuda.get_device_name(0)} | torch {torch.__version__}"
    f" | transformers {transformers.__version__}",
    flush=True,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", force=True)

from dataclasses import asdict  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

from gradienttrace.config import load_config  # noqa: E402
from gradienttrace.data import Sample, norm_text  # noqa: E402
from gradienttrace.evaluate import load_model_and_tokenizer, predict_probs  # noqa: E402
from gradienttrace.runtime import detect_runtime, resolve_device  # noqa: E402

GENRES = {
    "news": {
        "dataset": "fancyzhx/ag_news",
        "split": "test",
        "system": "You are a wire-service journalist.",
        "instruction": (
            "Write a news article about this story in about {words} words. Keep it plain and "
            "factual. Return only the article text, no headline and no preamble.\n\nStory: {snippet}"
        ),
    },
    "reviews": {
        "dataset": "Yelp/yelp_review_full",
        "split": "test",
        "system": "You write customer reviews.",
        "instruction": (
            "Write a {stars}-star customer review in about {words} words about this: {snippet}\n"
            "Keep it natural and specific. Return only the review text, no preamble."
        ),
    },
}
SNIPPET_CHARS = 200

cfg = load_config("configs/base.yaml")
runtime = detect_runtime()
device = resolve_device(runtime)
print(
    f"[gate] runtime={runtime.name} precision={runtime.precision}"
    f" pad_to_max={runtime.pad_to_max} device={device}",
    flush=True,
)

out_dir = Path("runs/gradientTrace")
out_dir.mkdir(parents=True, exist_ok=True)
gate_path = out_dir / f"{GATE_NAME}.jsonl"
meta_path = out_dir / f"{GATE_NAME}_meta.json"
report_path = out_dir / f"{GATE_NAME}_report_{MODEL_LABEL}.json"

api = None
if PUSH:
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ["HF_TOKEN"])

if REGENERATE and gate_path.is_file():
    gate_path.unlink()
if PUSH and not REGENERATE and not gate_path.is_file():
    try:  # reuse the frozen dataset instead of paying for generation again
        from huggingface_hub import hf_hub_download

        fetched = hf_hub_download(
            repo_id=GATE_REPO, filename=f"{GATE_NAME}.jsonl", token=os.environ["HF_TOKEN"]
        )
        gate_path.write_text(Path(fetched).read_text(encoding="utf-8"), encoding="utf-8")
        print("[gate] restored the frozen gate dataset from the Hub", flush=True)
    except Exception as exc:
        print("[gate] no frozen gate on the Hub yet:", repr(exc), flush=True)


def load_frozen(path):
    rows = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:  # a killed run can leave a half-written last line
                continue
            if isinstance(rec.get("id"), str) and isinstance(rec.get("text"), str) and rec.get("label") in (0, 1):
                rows[rec["id"]] = rec
    return rows


def make_snippet(text, limit):
    # The first "sentence" is often a dateline ("Ky.") or a one-liner ("Don't waste
    # your time."); keep adding sentences until the topic is substantive.
    parts = re.split(r"(?<=[.!?])\s+", text[:limit])
    snippet = ""
    for part in parts:
        if snippet and len(snippet) >= 60:
            break
        snippet = (snippet + " " + part).strip()
    return snippet


def load_humans():
    from datasets import load_dataset

    humans = []
    for genre, spec in GENRES.items():
        ds = load_dataset(spec["dataset"], split=spec["split"])
        picked = 0
        for row in ds:
            text = norm_text(row.get("text") or "")
            if len(text) < 80:  # a 5-word headline gives the generator nothing to work with
                continue
            stars = None
            if genre == "reviews":
                try:
                    stars = int(row.get("label")) + 1
                except (TypeError, ValueError):
                    stars = 3
            humans.append(
                {
                    "id": f"{genre}-{picked:04d}",
                    "pair_id": f"{genre}-{picked:04d}",
                    "genre": genre,
                    "label": 0,
                    "label_name": "human",
                    "text": text,
                    "generator": "human",
                    "source": spec["dataset"],
                    "prompt": make_snippet(text, SNIPPET_CHARS),
                    "stars": stars,
                    "human_words": len(text.split()),
                }
            )
            picked += 1
            if picked >= N_PER_GENRE:
                break
        print(f"[gate] {genre}: {picked} human samples from {spec['dataset']}", flush=True)
    return humans


def chat_prompt(tok, rec):
    spec = GENRES[rec["genre"]]
    instruction = spec["instruction"].format(
        words=max(20, min(int(rec["human_words"]), 260)),
        snippet=rec["prompt"],
        stars=rec.get("stars") or 3,
    )
    messages = [
        {"role": "system", "content": spec["system"]},
        {"role": "user", "content": instruction},
    ]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate_missing(humans, frozen):
    pending = [rec for rec in humans if f"{rec['id']}-ai" not in frozen]
    if not pending:
        print("[gate] all AI samples already generated", flush=True)
        return
    from transformers import AutoModelForCausalLM, AutoTokenizer

    stage(f"loading {GEN_MODEL} for {len(pending)} generations")
    tok = AutoTokenizer.from_pretrained(GEN_MODEL, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    gen_model = AutoModelForCausalLM.from_pretrained(GEN_MODEL, torch_dtype=torch.float16)
    gen_model.to(device)
    gen_model.eval()
    torch.manual_seed(0)
    try:
        with gate_path.open("a", encoding="utf-8") as fh:
            for start in range(0, len(pending), GEN_BATCH):
                batch = pending[start : start + GEN_BATCH]
                prompts = [chat_prompt(tok, rec) for rec in batch]
                enc = tok(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=1024,
                ).to(device)
                target_tokens = max(
                    len(tok.encode(rec["text"], add_special_tokens=False)) for rec in batch
                )
                max_new = min(640, int(target_tokens * 1.6) + 24)
                with torch.inference_mode():
                    out = gen_model.generate(
                        **enc,
                        max_new_tokens=max_new,
                        do_sample=True,
                        temperature=TEMPERATURE,
                        top_p=TOP_P,
                        pad_token_id=tok.eos_token_id,
                    )
                generated = out[:, enc["input_ids"].shape[1] :]
                for rec, ids in zip(batch, generated):
                    text = norm_text(tok.decode(ids, skip_special_tokens=True))
                    if ids[-1].item() != tok.eos_token_id:
                        # hit the length cap mid-sentence; cut back to the last full sentence
                        limit = int(len(rec["text"]) * 1.5)
                        if len(text) > limit:
                            window = text[:limit]
                            cut = max(window.rfind(". "), window.rfind("! "), window.rfind("? "))
                            text = window[: cut + 1].strip() if cut > limit // 2 else window.strip()
                    if not text:
                        continue
                    row = {
                        "id": f"{rec['id']}-ai",
                        "pair_id": rec["id"],
                        "genre": rec["genre"],
                        "label": 1,
                        "label_name": "ai",
                        "text": text,
                        "generator": GEN_MODEL,
                        "source": "generated",
                        "prompt": rec["prompt"],
                        "stars": rec.get("stars"),
                        "human_words": rec["human_words"],
                        "max_new_tokens": max_new,
                    }
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    frozen[row["id"]] = row
                fh.flush()
                done = min(start + GEN_BATCH, len(pending))
                if done % (GEN_BATCH * 10) == 0 or done == len(pending):
                    print(
                        f"[gate] generated {done}/{len(pending)} in {time.time() - T0:.0f}s",
                        flush=True,
                    )
                if PUSH and done % 200 == 0 and api is not None:
                    try:  # a Colab disconnect loses the VM disk; ship the partials
                        api.upload_file(
                            path_or_fileobj=str(gate_path),
                            path_in_repo=f"{GATE_NAME}.jsonl",
                            repo_id=GATE_REPO,
                            commit_message=f"gradientTrace gate partial: {done} AI samples",
                        )
                        print("[gate] partial gate dataset pushed to the Hub", flush=True)
                    except Exception as exc:
                        print("[gate] partial push failed (generation continues):", repr(exc), flush=True)
    finally:
        del gen_model
        import gc

        gc.collect()
        torch.cuda.empty_cache()


humans = load_humans()
frozen = load_frozen(gate_path)
with gate_path.open("a", encoding="utf-8") as fh:
    for rec in humans:
        if rec["id"] not in frozen:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            frozen[rec["id"]] = rec
print(f"[gate] frozen dataset rows: {len(frozen)} at {gate_path}", flush=True)

generate_missing(humans, frozen)
frozen = load_frozen(gate_path)

records = []
missing_pairs = 0
for rec in humans:
    if rec["id"] in frozen:
        records.append(frozen[rec["id"]])
    else:
        missing_pairs += 1
    ai_id = f"{rec['id']}-ai"
    if ai_id in frozen:
        records.append(frozen[ai_id])
    else:
        missing_pairs += 1
print(
    f"[gate] scoring {len(records)} rows ({missing_pairs} missing);"
    f" human {sum(r['label'] == 0 for r in records)},"
    f" ai {sum(r['label'] == 1 for r in records)}",
    flush=True,
)

meta = {
    "gate": GATE_NAME,
    "generator_model": GEN_MODEL,
    "temperature": TEMPERATURE,
    "top_p": TOP_P,
    "n_per_genre": N_PER_GENRE,
    "snippet_chars": SNIPPET_CHARS,
    "genres": {name: spec["dataset"] for name, spec in GENRES.items()},
    "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

stage(f"loading detector {MODEL_ID} ({MODEL_LABEL})")
detector, det_tok = load_model_and_tokenizer(MODEL_ID)
detector.to(device)
max_length = int(getattr(detector.config, "gradienttrace_max_length", cfg.max_length))
print(
    f"[gate] detector params {sum(p.numel() for p in detector.parameters()) / 1e6:.0f}M"
    f" | max_length {max_length}",
    flush=True,
)


def score(batch_size):
    samples = [
        Sample(
            text=rec["text"],
            label=int(rec["label"]),
            generator=rec["generator"],
            source=rec["source"],
            prompt=rec.get("prompt") or "",
            domain=rec["genre"],
        )
        for rec in records
    ]
    try:
        return predict_probs(
            detector,
            det_tok,
            samples,
            device,
            max_length=max_length,
            batch_size=batch_size,
            pad_to_max=runtime.pad_to_max,
            desc="gate",
        )
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if batch_size <= 8:
            raise
        print(f"[gate] OOM at batch {batch_size}; retrying at {batch_size // 2}", flush=True)
        return score(batch_size // 2)


probs = score(BATCH_SIZE)


def at_threshold(y, p, target):
    fpr, tpr, thresholds = roc_curve(y, p)
    idx = max(int(np.searchsorted(fpr, target, side="right") - 1), 0)
    return {"threshold": float(thresholds[idx]), "fpr": float(fpr[idx]), "tpr": float(tpr[idx])}


LENGTH_EDGES = (0, 400, 800, 1200, 10**9)


def pair_metrics(recs, ps):
    rec_by_id = {rec["id"]: rec for rec in recs}
    prob_by_id = {rec["id"]: float(prob) for rec, prob in zip(recs, ps)}
    pairs = []
    for rec in recs:
        ai_rec = rec_by_id.get(f"{rec['id']}-ai")
        if rec["label"] == 0 and ai_rec is not None:
            pairs.append(
                (
                    prob_by_id[rec["id"]],
                    prob_by_id[ai_rec["id"]],
                    len(rec["text"]),
                    len(ai_rec["text"]),
                )
            )
    out = {
        "n_pairs": len(pairs),
        "pairwise_acc": float(np.mean([a > h for h, a, _, _ in pairs])) if pairs else None,
    }
    matched = [(h, a) for h, a, lh, la in pairs if lh > 0 and 0.66 <= la / lh <= 1.5]
    out["n_matched"] = len(matched)
    out["matched_auroc"] = (
        float(roc_auc_score([0] * len(matched) + [1] * len(matched),
                            [h for h, _ in matched] + [a for _, a in matched]))
        if len(matched) >= 5
        else None
    )
    return out


def bucket_metrics(recs, ps):
    buckets = []
    for lo, hi in zip(LENGTH_EDGES, LENGTH_EDGES[1:]):
        human = [p for rec, p in zip(recs, ps) if rec["label"] == 0 and lo <= len(rec["text"]) < hi]
        ai = [p for rec, p in zip(recs, ps) if rec["label"] == 1 and lo <= len(rec["text"]) < hi]
        entry = {
            "chars": f"{lo}-{'+' if hi == 10**9 else hi}",
            "n_human": len(human),
            "n_ai": len(ai),
            "mean_human": float(np.mean(human)) if human else None,
            "mean_ai": float(np.mean(ai)) if ai else None,
        }
        if len(human) >= 5 and len(ai) >= 5:
            entry["auroc"] = float(
                roc_auc_score([0] * len(human) + [1] * len(ai), human + ai)
            )
        buckets.append(entry)
    return buckets


def genre_metrics(recs, ps):
    human = [p for rec, p in zip(recs, ps) if rec["label"] == 0]
    ai = [p for rec, p in zip(recs, ps) if rec["label"] == 1]
    out = {
        "n_human": len(human),
        "n_ai": len(ai),
        "mean_human": float(np.mean(human)) if human else None,
        "mean_ai": float(np.mean(ai)) if ai else None,
        "human_p50": float(np.percentile(human, 50)) if human else None,
        "human_p90": float(np.percentile(human, 90)) if human else None,
        "human_p99": float(np.percentile(human, 99)) if human else None,
        "fpr_at_0_5": float(np.mean(np.asarray(human) > 0.5)) if human else None,
        "tpr_at_0_5": float(np.mean(np.asarray(ai) > 0.5)) if ai else None,
    }
    if len(human) >= 5 and len(ai) >= 5:
        y = np.array([0] * len(human) + [1] * len(ai))
        p = np.array(human + ai)
        out["auroc"] = float(roc_auc_score(y, p))
        out["at_1pct_fpr"] = at_threshold(y, p, 0.01)
        out["at_5pct_fpr"] = at_threshold(y, p, 0.05)
    out.update(pair_metrics(recs, ps))
    out["length_buckets"] = bucket_metrics(recs, ps)
    return out


report = {
    "model": MODEL_ID,
    "model_label": MODEL_LABEL,
    "gate": GATE_NAME,
    "generator_model": GEN_MODEL,
    "runtime": asdict(runtime),
    "n_rows": len(records),
    "missing_pairs": missing_pairs,
    "metrics": {
        **{name: genre_metrics([r for r in records if r["genre"] == name],
                               [p for r, p in zip(records, probs) if r["genre"] == name])
           for name in GENRES},
        "pooled": genre_metrics(records, probs),
    },
}
report_path.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")


def fmt4(value):
    try:
        return format(float(value), ".4f")
    except (TypeError, ValueError):
        return "n/a"


print(f"\n[gate] {'slice':10s} {'n_h':>5s} {'n_ai':>5s} {'auroc':>8s} {'tpr@1%':>7s}"
      f" {'fpr@0.5':>8s} {'pair':>6s} {'matched':>8s}", flush=True)
for name, metrics in report["metrics"].items():
    t1 = (metrics.get("at_1pct_fpr") or {}).get("tpr")
    print(
        f"[gate] {name:10s} {metrics['n_human']:5d} {metrics['n_ai']:5d}"
        f" {fmt4(metrics.get('auroc')):>8s} {fmt4(t1):>7s}"
        f" {fmt4(metrics.get('fpr_at_0_5')):>8s} {fmt4(metrics.get('pairwise_acc')):>6s}"
        f" {fmt4(metrics.get('matched_auroc')):>8s}",
        flush=True,
    )
print(
    "[gate] read TPR@1% first: it is the AI caught when the human pool is allowed a 1% false-positive rate",
    flush=True,
)

if PUSH and api is not None:
    for path, dest, message in (
        (gate_path, f"{GATE_NAME}.jsonl", "gradientTrace frozen external gate dataset"),
        (meta_path, f"{GATE_NAME}_meta.json", "gradientTrace external gate metadata"),
        (report_path, f"{GATE_NAME}_report_{MODEL_LABEL}.json", f"gradientTrace external gate report ({MODEL_LABEL})"),
    ):
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=dest,
            repo_id=GATE_REPO,
            commit_message=message,
        )
    print(
        f"[gate] pushed {GATE_NAME}.jsonl, {GATE_NAME}_meta.json,"
        f" {GATE_NAME}_report_{MODEL_LABEL}.json -> https://huggingface.co/{GATE_REPO}",
        flush=True,
    )

print(f"\n[gate] done in {time.time() - T0:.0f}s", flush=True)

# keep GPU 0 clean: a training cell in this same kernel would otherwise OOM at step 0
import gc  # noqa: E402

del detector
gc.collect()
torch.cuda.empty_cache()
