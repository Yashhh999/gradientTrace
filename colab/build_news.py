# gradientTrace - build paired clean-news data (paste this whole file into one Kaggle cell)
# Kaggle: Accelerator = GPU T4 x2, Internet = On. Run it before the training cell.
# v2's RAID news/reddit/books/poetry humans are 100% perturbed while its AI rows are clean, so
# the model learned "clean prose = AI" and flagged clean human news at the gate (news AUROC
# 0.4775). This cell adds the missing contrast: clean human news leads paired with AI news on
# the same story, written with the external gate's own journalist prompt, plus human-only wire
# stories from vblagoje/cc_news (title + description, the gate's headline+body form). CNN leads
# alone did not transfer: the gate still scored clean ag_news humans at mean P(AI) 0.918.
# It also writes short variants (first sentence / short wire brief, 100-240 chars): the gate's
# Qwen news is 1.3 sentences (p50 134 chars) while every generated AI row was 220+ chars, so
# length alone separated the classes. Every CNN lead and AI text gets a paired first-sentence
# variant so short news is not implicitly AI, and cc_news contributes published short briefs.
# Output: v2/news_pairs.jsonl on the Hub, restored by colab/one_shot_kaggle.py into
# data/local.jsonl.
HUB_ID = "Yashhh999/gradientTrace"
DATASET = "nschantz21/cnn_dailymail-parsed"  # parquet copy of CNN/DailyMail 3.0.0
SPLIT = "train"
N_STORIES = 4000              # human leads; each is paired with one text per generator
EXTRA_HUMAN_DATASET = "vblagoje/cc_news"  # title + description wire stories, many outlets
EXTRA_HUMAN_CONFIG = "plain_text"
EXTRA_HUMAN_SPLIT = "train"
N_EXTRA_HUMANS = 6000         # human-only rows; the 4,000 CNN leads cover only 20% of the bucket
N_SHORT_HUMANS = 6000         # cc_news briefs; with the CNN first sentences this matches the AI short share
SHORT_LEAD_CHARS = 100
SHORT_MIN = 100               # one gate-length sentence; data.min_chars is 100 in v2/pilot
SHORT_MAX = 240               # gate AI p95 is 223 chars
TARGET_GENERATORS = 2
GEN_MODELS = [
    "microsoft/Phi-3-mini-4k-instruct",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",  # Granite-2B produced NaN logits in fp16 on T4
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


def make_wire_lead(row, min_chars=MIN_LEAD_CHARS, max_chars=MAX_LEAD_CHARS, cut=True):
    """cc_news title + description: the gate's ag_news humans are headline + wire description."""
    title = norm(row.get("title") or "")
    desc = norm(row.get("description") or "")
    body = norm(row.get("text") or "")
    head = title if title.endswith((".", "!", "?", ":")) else (title + "." if title else "")
    if len(head) > max_chars:
        return None
    text = (head + " " + desc).strip()
    if len(text) < min_chars and body:
        for part in _SENT.split(body):
            candidate = (text + " " + part.strip()).strip()
            if len(candidate) > max_chars:
                break
            text = candidate
            if len(text) >= min_chars:
                break
    if len(text) > max_chars:
        if not cut:  # a short brief must be short as published, never a prefix of a long row
            return None
        text = cut_to_sentence(text, max_chars)
    if not text:
        return None
    if not (min_chars <= len(text) <= max_chars):
        return None
    if not text.endswith((".", "!", "?")):
        return None
    return text


def make_short(text):
    """First sentence of a news text, the gate's 1-3 sentence output shape."""
    ends = [match.end() for match in _END.finditer(text)]
    if not ends:
        return None
    cut = text[: ends[0]].strip()
    if len(cut) < SHORT_MIN and len(ends) > 1:
        cut = text[: ends[1]].strip()
    if not (SHORT_MIN <= len(cut) <= SHORT_MAX):
        return None
    return cut


def text_key(text):
    return hashlib.sha1(norm(text).lower().encode("utf-8", "ignore")).hexdigest()


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


REJECTS = Counter()
REJECT_EXAMPLES = {}


def reject(reason, text):
    REJECTS[reason] += 1
    samples = REJECT_EXAMPLES.setdefault(reason, [])
    if len(samples) < 2 and text:
        samples.append(norm(text)[:240])
    return None


def clean_output(text, human, ended):
    text = norm(text)
    if not text:
        return reject("empty", "")
    low = text.lower()
    if low.startswith(_BAD_STARTS):
        return reject("refusal", text)
    if any(phrase in low[:200] for phrase in _BAD_MENTIONS):
        return reject("preamble", text)
    limit = int(len(human) * 1.6)
    cutoff = text
    if not ended or len(text) > limit:
        text = cut_to_sentence(text, limit)
    if not text:
        return reject("no_sentence", cutoff)
    if len(text) < MIN_AI_CHARS:
        return reject("short", text)
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


def load_extra_humans(count, exclude, seen, min_chars=MIN_LEAD_CHARS, max_chars=MAX_LEAD_CHARS, cut=True):
    from datasets import load_dataset

    stage(f"reading {count} more human wire stories from {EXTRA_HUMAN_DATASET}")
    ds = load_dataset(
        EXTRA_HUMAN_DATASET, EXTRA_HUMAN_CONFIG, split=EXTRA_HUMAN_SPLIT, streaming=True
    )
    stories = []
    scanned = 0
    for row in ds:
        scanned += 1
        lead = make_wire_lead(row, min_chars, max_chars, cut)
        if lead is None:
            continue
        key = text_key(lead)
        if key in seen:
            continue
        seen.add(key)
        raw_id = str(row.get("url") or key)
        group = "cc-" + hashlib.sha1(raw_id.encode("utf-8", "ignore")).hexdigest()[:12]
        if group in exclude:
            continue
        stories.append({"group": group, "text": lead, "prompt": make_snippet(lead)})
        if len(stories) >= count:
            break
        if scanned % 20000 == 0:
            print(
                f"[news] cc_news scanned {scanned} -> {len(stories)} usable rows"
                f" | {time.time() - T0:.0f}s",
                flush=True,
            )
    stage(f"cc_news scanned {scanned} -> {len(stories)} usable rows")
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


def extra_human_row(story, variant="raw"):
    return {
        "text": story["text"],
        "label": 0,
        "generator": "human",
        "source": EXTRA_HUMAN_DATASET,
        "prompt": story["prompt"],
        "domain": "news",
        "variant": variant,
        "bucket": "human",
        "group": story["group"],
    }


def interleave(groups):
    """Alternate rows from each list so a pool cap cannot take only one variant."""
    pools = [list(group) for group in groups if group]
    out = []
    while pools:
        for pool in pools:
            out.append(pool.pop(0))
        pools = [pool for pool in pools if pool]
    return out


def add_short_variants():
    """Pair each CNN lead and AI text with a gate-length first sentence, then interleave."""
    rows = load_rows()
    have_short = {
        (rec["label"], rec["group"]) for rec in rows if rec.get("variant") == "short"
    }
    expanded = []
    added_ai = 0
    added_human = 0
    for rec in rows:
        expanded.append(rec)
        if (rec["label"], rec["group"]) in have_short:
            continue
        if rec["label"] == 0 and rec.get("source") != DATASET:
            continue  # cc_news briefs come from their own stage; long cc rows have no AI pair
        text = make_short(rec["text"])
        if text is None or text == rec["text"]:
            continue
        short = dict(rec)
        short["text"] = text
        short["variant"] = "short"
        expanded.append(short)
        if rec["label"] == 1:
            added_ai += 1
        else:
            added_human += 1
    humans = [rec for rec in expanded if rec["label"] == 0]
    ai = [rec for rec in expanded if rec["label"] == 1]
    ordered = interleave(
        [
            [rec for rec in humans if rec.get("variant") == "short"],
            [rec for rec in humans if rec.get("variant") != "short"],
        ]
    ) + interleave(
        [
            [rec for rec in ai if rec.get("variant") == "short"],
            [rec for rec in ai if rec.get("variant") != "short"],
        ]
    )
    tmp = OUT.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for rec in ordered:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp.replace(OUT)
    return added_ai, added_human


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
    examples_shown = False
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
                    # Phi-3 often ends after ~25 words; force roughly the human's length so the
                    # 200-char filter does not discard most of the corpus.
                    min_new_tokens=min(max_new, max(48, target_tokens)),
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
                print(f"[news] rejects: {dict(REJECTS)}", flush=True)
            if done >= 80 and not examples_shown:
                examples_shown = True
                for reason, samples in REJECT_EXAMPLES.items():
                    print(f"[news] reject example [{reason}]: {samples[0]!r}", flush=True)
            if written - last_push >= PUSH_EVERY:
                last_push = written
                try:  # a disconnected VM loses its disk; ship the partials
                    fh.flush()
                    push([(OUT, "v2/news_pairs.jsonl")], f"gradientTrace news pairs: {last_push} AI texts")
                except Exception as exc:
                    print("[news] partial push failed (generation continues):", repr(exc), flush=True)
    finally:
        try:  # a later model can crash the CUDA context; never lose texts already generated
            fh.flush()
            push([(OUT, "v2/news_pairs.jsonl")], f"gradientTrace news pairs: {written} new AI texts")
        except Exception as exc:
            print("[news] final push failed:", repr(exc), flush=True)
        del model
        import gc

        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception as exc:  # a poisoned context raises here; keep the original traceback
            print("[news] cuda cleanup skipped:", repr(exc), flush=True)
    stage(f"{model_id}: wrote {written}/{len(pending)} AI texts")
    return True


restore()
rows = load_rows()
stories = {}
for rec in rows:
    if rec["label"] == 0 and rec.get("variant") != "short" and rec["group"] not in stories:
        if rec.get("source") == EXTRA_HUMAN_DATASET:  # human-only rows: never paired, never generated
            continue
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

    existing = load_rows()
    extra_have = sum(
        1
        for rec in existing
        if rec["label"] == 0
        and rec.get("source") == EXTRA_HUMAN_DATASET
        and rec.get("variant") != "short"
    )
    if extra_have < N_EXTRA_HUMANS:
        extra = load_extra_humans(
            N_EXTRA_HUMANS - extra_have,
            {rec["group"] for rec in existing if rec["label"] == 0},
            {text_key(rec["text"]) for rec in existing},
        )
        for story in extra:
            fh.write(json.dumps(extra_human_row(story), ensure_ascii=False) + "\n")
        fh.flush()
        if extra:
            push(
                [(OUT, "v2/news_pairs.jsonl")],
                f"gradientTrace news pairs: {extra_have + len(extra)} cc_news human rows",
            )
    else:
        print(f"[news] cc_news: {extra_have} human rows already present", flush=True)

    existing = load_rows()
    short_have = sum(
        1
        for rec in existing
        if rec["label"] == 0
        and rec.get("source") == EXTRA_HUMAN_DATASET
        and rec.get("variant") == "short"
    )
    if short_have < N_SHORT_HUMANS:
        short = load_extra_humans(
            N_SHORT_HUMANS - short_have,
            {rec["group"] for rec in existing if rec["label"] == 0},
            {text_key(rec["text"]) for rec in existing},
            SHORT_LEAD_CHARS,
            MIN_LEAD_CHARS - 1,
            False,
        )
        for story in short:
            fh.write(json.dumps(extra_human_row(story, "short"), ensure_ascii=False) + "\n")
        fh.flush()
        if short:
            push(
                [(OUT, "v2/news_pairs.jsonl")],
                f"gradientTrace news pairs: {short_have + len(short)} short cc_news rows",
            )
    else:
        print(f"[news] cc_news short: {short_have} rows already present", flush=True)

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

short_ai_added, short_human_added = add_short_variants()
if short_ai_added or short_human_added:
    stage(
        f"wrote {short_human_added} short human leads and {short_ai_added} short AI variants"
        " and rebalanced the file"
    )
    push(
        [(OUT, "v2/news_pairs.jsonl")],
        f"gradientTrace news pairs: {short_ai_added} short AI, {short_human_added} short human",
    )

rows = load_rows()
humans = [rec for rec in rows if rec["label"] == 0]
ai = [rec for rec in rows if rec["label"] == 1]
short_humans = [rec for rec in humans if rec.get("variant") == "short"]
short_ai = [rec for rec in ai if rec.get("variant") == "short"]
long_ai = [rec for rec in ai if rec.get("variant") != "short"]
by_model = Counter(rec["generator"] for rec in long_ai)
covered = {rec["group"] for rec in ai}
# cc_news rows are human-only, so they never count as a story needing an AI pair.
cnn_groups = {
    rec["group"] for rec in humans if rec.get("source") != EXTRA_HUMAN_DATASET
}
meta = {
    "dataset": DATASET,
    "split": SPLIT,
    "extra_human_dataset": EXTRA_HUMAN_DATASET,
    "n_stories": len(cnn_groups),
    "generators": sorted(by_model),
    "temperature": TEMPERATURE,
    "top_p": TOP_P,
    "system": SYSTEM,
    "instruction": INSTRUCTION,
    "lead_chars": [MIN_LEAD_CHARS, MAX_LEAD_CHARS],
    "short_chars": [SHORT_MIN, SHORT_MAX],
    "snippet_chars": SNIPPET_CHARS,
    "counts": {
        "human": len(humans),
        "human_short": len(short_humans),
        "human_short_sources": dict(Counter(rec.get("source") for rec in short_humans)),
        "human_sources": dict(Counter(rec.get("source") for rec in humans)),
        "ai": len(ai),
        "ai_short": len(short_ai),
        "ai_by_model": dict(by_model),
        "stories_with_ai": len(cnn_groups & covered),
        "stories_without_ai": len(cnn_groups - covered),
    },
    "lengths": {
        "human": length_stats([len(rec["text"]) for rec in humans]),
        "human_short": length_stats([len(rec["text"]) for rec in short_humans]),
        "ai_short": length_stats([len(rec["text"]) for rec in short_ai]),
        **{
            f"ai/{model}": length_stats(
                [len(rec["text"]) for rec in long_ai if rec["generator"] == model]
            )
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

print(f"\n[news] human {len(humans)} {dict(Counter(rec.get('source') for rec in humans))}", flush=True)
print(
    f"[news] human short {len(short_humans)}"
    f" {dict(Counter(rec.get('source') for rec in short_humans))}",
    flush=True,
)
print(f"[news] ai {len(ai)} ({len(short_ai)} short) {dict(by_model)}", flush=True)
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
