# gradientTrace - build paired clean-news data (paste this whole file into one Kaggle cell)
# Kaggle: Accelerator = GPU T4 x2, Internet = On. Run it before the training cell.
# v2's RAID news/reddit/books/poetry humans are 100% perturbed while its AI rows are clean, so
# the model learned "clean prose = AI" and flagged clean human news at the gate (news AUROC
# 0.4775). This cell adds the missing contrast: clean human news leads paired with AI news on
# the same story, written with the external gate's own journalist prompt. Output:
# v2/news_pairs.jsonl on the Hub, restored by colab/one_shot_kaggle.py into data/local.jsonl.
HUB_ID = "Yashhh999/gradientTrace"
DATASET = "nschantz21/cnn_dailymail-parsed"  # parquet copy of CNN/DailyMail 3.0.0
SPLIT = "train"
N_STORIES = 4000              # human leads; each is paired with one text per generator
TARGET_GENERATORS = 2
GEN_MODELS = [
    "microsoft/Phi-3-mini-4k-instruct",
    "ibm-granite/granite-3.0-2b-instruct",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",  # fallback if a newer architecture is unsupported
]
GEN_BATCH = 8
TEMPERATURE = 0.8
TOP_P = 0.9
MIN_LEAD_CHARS = 200          # a bare headline gives the generator nothing to work with
MAX_LEAD_CHARS = 520
MIN_AI_CHARS = 200            # data.min_chars: shorter rows would be dropped by the loader
SNIPPET_CHARS = 200           # same topic window as the external gate
PUSH_EVERY = 200              # generations between partial Hub pushes
REGENERATE = False            # True wipes the local and pushed dataset and starts over
# The gate's news prompt, verbatim; training pairs must be asked the same question as the gate.
SYSTEM = "You are a wire-service journalist."
INSTRUCTION = (
    "Write a news article about this story in about {words} words. Keep it plain and "
    "factual. Return only the article text, no headline and no preamble.\n\nStory: {snippet}"
)

import hashlib
import json
import os
import re
import socket
import time
from collections import Counter
from pathlib import Path

T0 = time.time()
OUT = Path("runs/v2/news_pairs.jsonl")
META_PATH = Path("runs/v2/news_pairs_meta.json")
OUT.parent.mkdir(parents=True, exist_ok=True)
_WS = re.compile(r"\s+")
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"“'(])")
_END = re.compile(r"[.!?](?:\s|$)")
_BAD_STARTS = ("i cannot", "i can't", "i'm sorry", "i am sorry", "as an ai")
_BAD_MENTIONS = ("here is the article", "here's the article", "here is a news article")


def stage(message):
    print(f"\n[news +{time.time() - T0:5.0f}s] {message}", flush=True)


def norm(text):
    return _WS.sub(" ", text or "").strip()


token = os.environ.get("HF_TOKEN") or ""
if not token:  # never getpass: it blocks a notebook cell
    try:
        from kaggle_secrets import UserSecretsClient

        token = UserSecretsClient().get_secret("HF_TOKEN") or ""
    except Exception as exc:
        print("[news] Kaggle secret unavailable:", exc, flush=True)
if not token:
    try:
        from google.colab import userdata

        token = userdata.get("HF_TOKEN") or ""
    except Exception as exc:
        print("[news] Colab secret unavailable:", exc, flush=True)
if not token:
    raise SystemExit(
        "[news] HF_TOKEN missing. Kaggle: Add-ons > Secrets, add HF_TOKEN (write scope) and "
        "enable it for this notebook; Colab: key icon in the left sidebar. Then re-run."
    )
os.environ["HF_TOKEN"] = token
print("[news] HF_TOKEN present (value not printed)", flush=True)

try:
    socket.create_connection(("huggingface.co", 443), timeout=10).close()
except OSError as exc:
    raise SystemExit("[news] no internet: Kaggle Settings > Internet > On, then re-run.") from exc

import importlib.util  # noqa: E402

missing = [
    package
    for module, package in (
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("datasets", "datasets"),
        ("sentencepiece", "sentencepiece"),
    )
    if importlib.util.find_spec(module) is None
]
if missing:
    stage(f"installing missing packages: {missing}")
    import subprocess
    import sys

    subprocess.run([sys.executable, "-m", "pip", "-q", "install", *missing], check=True)

import torch  # noqa: E402
import transformers  # noqa: E402
from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

if not torch.cuda.is_available():
    raise SystemExit("[news] no GPU: Settings > Accelerator > GPU T4 x2, then re-run")
print(
    f"[news] GPU {torch.cuda.get_device_name(0)} | torch {torch.__version__}"
    f" | transformers {transformers.__version__}",
    flush=True,
)
device = "cuda:0"
api = HfApi(token=token)


def make_lead(article):
    """The first sentences of a CNN/DailyMail article: clean, news-style human prose."""
    text = norm(article)
    if not text:
        return None
    lead = ""
    for part in _SENT.split(text):
        part = part.strip()
        if not part:
            continue
        candidate = (lead + " " + part).strip()
        if len(candidate) > MAX_LEAD_CHARS:
            break
        lead = candidate
        if len(lead) >= MIN_LEAD_CHARS:
            break
    if not (MIN_LEAD_CHARS <= len(lead) <= MAX_LEAD_CHARS):
        return None
    if not lead.endswith((".", "!", "?")):
        return None
    if "»" in lead or "E-mail to a friend" in lead:
        return None
    return lead


def make_snippet(text):
    # Mirrors colab/external_gate.py: keep adding sentences until the topic is substantive.
    parts = _SENT.split(text[:SNIPPET_CHARS])
    snippet = ""
    for part in parts:
        if snippet and len(snippet) >= 60:
            break
        snippet = (snippet + " " + part).strip()
    return snippet


def cut_to_sentence(text, limit):
    window = text[:limit]
    ends = [match.end() for match in _END.finditer(window)]
    if not ends:
        return None
    return window[: ends[-1]].strip()


def clean_output(text, human, ended):
    text = norm(text)
    if not text:
        return None
    low = text.lower()
    if low.startswith(_BAD_STARTS) or any(phrase in low[:200] for phrase in _BAD_MENTIONS):
        return None
    limit = int(len(human) * 1.6)
    if not ended or len(text) > limit:
        text = cut_to_sentence(text, limit)
    if not text or len(text) < MIN_AI_CHARS:
        return None
    return text


def length_stats(values):
    if not values:
        return {}
    values = sorted(values)
    n = len(values)
    return {
        "n": n,
        "p10": values[int(n * 0.10)],
        "p50": values[int(n * 0.50)],
        "p90": values[min(n - 1, int(n * 0.90))],
    }


def load_rows(path=OUT):
    rows = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            isinstance(rec.get("text"), str)
            and rec.get("label") in (0, 1)
            and isinstance(rec.get("group"), str)
            and rec["group"]
        ):
            rows.append(rec)
    return rows


def restore():
    if REGENERATE:
        for path in (OUT, META_PATH):
            if path.is_file():
                path.unlink()
        return
    if OUT.is_file():
        return
    try:
        fetched = hf_hub_download(
            repo_id=HUB_ID, filename="v2/news_pairs.jsonl", token=token
        )
        OUT.write_text(Path(fetched).read_text(encoding="utf-8"), encoding="utf-8")
        print("[news] restored the pushed pairs from the Hub", flush=True)
    except Exception as exc:
        print("[news] no pushed pairs to restore:", repr(exc), flush=True)


def push(files, message):
    for path, dest in files:
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=dest,
            repo_id=HUB_ID,
            commit_message=message,
        )
    print("[news] pushed", ", ".join(dest for _, dest in files), flush=True)


def load_stories(count, exclude):
    from datasets import load_dataset

    stage(f"reading {count} more leads from {DATASET} [{SPLIT}]")
    ds = load_dataset(DATASET, split=SPLIT, streaming=True)
    stories = []
    seen = set()
    scanned = 0
    for row in ds:
        scanned += 1
        lead = make_lead(row.get("article") or "")
        if lead is None:
            continue
        key = hashlib.sha1(lead.lower().encode("utf-8", "ignore")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        raw_id = str(row.get("id") or key)
        group = "cnn-" + hashlib.sha1(raw_id.encode()).hexdigest()[:12]
        if group in exclude:  # a resumed run must not re-read the leads it already has
            continue
        stories.append({"group": group, "text": lead, "prompt": make_snippet(lead)})
        if len(stories) >= count:
            break
        if scanned % 5000 == 0:
            print(
                f"[news] scanned {scanned} articles -> {len(stories)} usable leads"
                f" | {time.time() - T0:.0f}s",
                flush=True,
            )
    stage(f"scanned {scanned} articles -> {len(stories)} usable leads")
    return stories


def human_row(story):
    return {
        "text": story["text"],
        "label": 0,
        "generator": "human",
        "source": DATASET,
        "prompt": story["prompt"],
        "domain": "news",
        "variant": "raw",
        "bucket": "human",
        "group": story["group"],
    }


def ai_prompt(tok, story):
    instruction = INSTRUCTION.format(
        words=max(20, min(len(story["text"].split()), 260)),
        snippet=story["prompt"],
    )
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": instruction},
    ]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate_model(model_id, pending, stories, fh, index):
    """Generate one text per pending story; returns True if the model loaded and ran."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    stage(f"loading {model_id} for {len(pending)} generations")
    try:
        tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
        model.to(device)
        model.eval()
    except Exception as exc:
        print(f"[news] {model_id} unavailable, trying the next model:", repr(exc), flush=True)
        return False

    torch.manual_seed(1000 + index)
    written = 0
    last_push = 0
    model_t0 = time.time()
    try:
        for start in range(0, len(pending), GEN_BATCH):
            batch = pending[start : start + GEN_BATCH]
            prompts = [ai_prompt(tok, stories[group]) for group in batch]
            enc = tok(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,
            ).to(device)
            target_tokens = max(
                len(tok.encode(stories[group]["text"], add_special_tokens=False))
                for group in batch
            )
            max_new = min(400, int(target_tokens * 1.6) + 24)
            with torch.inference_mode():
                out = model.generate(
                    **enc,
                    max_new_tokens=max_new,
                    do_sample=True,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    pad_token_id=tok.eos_token_id,
                )
            generated = out[:, enc["input_ids"].shape[1] :]
            for group, ids in zip(batch, generated):
                story = stories[group]
                text = clean_output(
                    tok.decode(ids, skip_special_tokens=True),
                    story["text"],
                    ids[-1].item() == tok.eos_token_id,
                )
                if text is None:
                    continue
                fh.write(
                    json.dumps(
                        {
                            "text": text,
                            "label": 1,
                            "generator": model_id,
                            "source": "generated",
                            "prompt": story["prompt"],
                            "domain": "news",
                            "variant": "raw",
                            "bucket": "ai_foundational",
                            "group": group,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                written += 1
            fh.flush()
            done = min(start + GEN_BATCH, len(pending))
            if (start // GEN_BATCH) % 10 == 9 or done == len(pending):
                rate = (time.time() - model_t0) / max(written, 1)
                print(
                    f"[news] {model_id.split('/')[-1]}: {done}/{len(pending)} stories,"
                    f" {written} texts kept, {rate:.2f}s/text | {time.time() - T0:.0f}s",
                    flush=True,
                )
            if written - last_push >= PUSH_EVERY:
                last_push = written
                try:  # a disconnected VM loses its disk; ship the partials
                    fh.flush()
                    push([(OUT, "v2/news_pairs.jsonl")], f"gradientTrace news pairs: {last_push} AI texts")
                except Exception as exc:
                    print("[news] partial push failed (generation continues):", repr(exc), flush=True)
    finally:
        del model
        import gc

        gc.collect()
        torch.cuda.empty_cache()
    stage(f"{model_id}: wrote {written}/{len(pending)} AI texts")
    return True


restore()
rows = load_rows()
stories = {}
for rec in rows:
    if rec["label"] == 0 and rec["group"] not in stories:
        stories[rec["group"]] = {
            "text": rec["text"],
            "prompt": rec.get("prompt") or make_snippet(rec["text"]),
            "group": rec["group"],
        }
print(f"[news] existing pairs file: {len(rows)} rows, {len(stories)} stories", flush=True)

with OUT.open("a", encoding="utf-8") as fh:
    if len(stories) < N_STORIES:
        for story in load_stories(N_STORIES - len(stories), stories):
            fh.write(json.dumps(human_row(story), ensure_ascii=False) + "\n")
            stories[story["group"]] = story
        fh.flush()
        if len(stories):
            push([(OUT, "v2/news_pairs.jsonl")], f"gradientTrace news pairs: {len(stories)} human leads")
    if not stories:
        raise SystemExit(
            "[news] no usable human leads; check that the dataset split still has an 'article' column"
        )
    print(f"[news] stories ready: {len(stories)}", flush=True)

    chosen = []
    for index, model_id in enumerate(GEN_MODELS):
        if len(chosen) >= TARGET_GENERATORS:
            break
        have = {
            rec["group"]
            for rec in load_rows()
            if rec["label"] == 1 and rec["generator"] == model_id
        }
        pending = [group for group in stories if group not in have]
        if not pending:
            if have:
                print(f"[news] {model_id}: all stories already covered", flush=True)
                chosen.append(model_id)
            continue
        if generate_model(model_id, pending, stories, fh, index):
            chosen.append(model_id)

    if not chosen:
        raise SystemExit("[news] no generator model could be loaded; nothing was paired")

rows = load_rows()
humans = [rec for rec in rows if rec["label"] == 0]
ai = [rec for rec in rows if rec["label"] == 1]
by_model = Counter(rec["generator"] for rec in ai)
covered = {rec["group"] for rec in ai}
meta = {
    "dataset": DATASET,
    "split": SPLIT,
    "n_stories": len(humans),
    "generators": sorted(by_model),
    "temperature": TEMPERATURE,
    "top_p": TOP_P,
    "system": SYSTEM,
    "instruction": INSTRUCTION,
    "lead_chars": [MIN_LEAD_CHARS, MAX_LEAD_CHARS],
    "snippet_chars": SNIPPET_CHARS,
    "counts": {
        "human": len(humans),
        "ai": len(ai),
        "ai_by_model": dict(by_model),
        "stories_with_ai": len(covered),
        "stories_without_ai": len({rec["group"] for rec in humans} - covered),
    },
    "lengths": {
        "human": length_stats([len(rec["text"]) for rec in humans]),
        **{
            f"ai/{model}": length_stats([len(rec["text"]) for rec in ai if rec["generator"] == model])
            for model in sorted(by_model)
        },
    },
    "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
push(
    [(OUT, "v2/news_pairs.jsonl"), (META_PATH, "v2/news_pairs_meta.json")],
    f"gradientTrace news pairs: {meta['n_stories']} stories, {meta['counts']['ai']} AI texts",
)

print(f"\n[news] human {len(humans)}, ai {len(ai)} {dict(by_model)}", flush=True)
print(f"[news] lengths: {json.dumps(meta['lengths'])}", flush=True)
free_gib = torch.cuda.mem_get_info(0)[0] / 2**30
print(
    f"[news] done in {time.time() - T0:.0f}s -> {HUB_ID}/v2/news_pairs.jsonl"
    f" | GPU 0 free {free_gib:.1f} GiB",
    flush=True,
)
if free_gib < 13.0:  # the training cell aborts below this; a restart is cheaper than a failed run
    print(
        "[news] GPU memory is still held by this kernel: restart the session before the training cell.",
        flush=True,
    )
