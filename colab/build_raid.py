# gradientTrace - build the RAID slice used by v2 training (paste this whole file into one cell)
# Kaggle: Internet > On. A GPU is not needed; run it before the training cell.
# RAID train.csv is one 11.8 GB CSV whose domains are stored in contiguous blocks. This cell
# reads only the byte ranges for the domains v2 needs (HTTP Range requests), keeps paired
# human/model rows plus paraphrase/synonym humanised AI, and pushes the result to the Hub as
# v2/raid_v2.jsonl. The run is resumable: each finished domain is added to the pushed file.
HUB_ID = "Yashhh999/gradientTrace"   # repo that stores the extracted slice
RAID_URL = "https://huggingface.co/datasets/liamdugan/raid/resolve/main/train.csv"
# (start GB, end GB) measured on train.csv; each block starts on a row boundary near these
WINDOWS = {  # byte ranges probed on train.csv; overlaps are filtered out by the domain column
    "news":    (3.30, 5.10),
    "reviews": (9.04, 9.93),
    "reddit":  (7.85, 9.05),
    "wiki":    (9.93, 11.78),
    "books":   (1.75, 3.35),
    "recipes": (6.20, 7.90),
    "poetry":  (5.05, 6.22),
}
HUMAN_CAP = 3000        # per domain
AI_CAP = 6000           # per domain, attack == none
MODEL_CAP = 900         # per domain per model, keeps one model from filling AI_CAP
HUMANIZED_CAP = 1500    # per domain, paraphrase/synonym attacks on model text
HUMANIZED_MODEL_CAP = 300
HUMAN_ATTACK_CAP = 1500 # per domain per attack; RAID also perturbs human text (spelling,
                        # whitespace, ...), so keep a mix instead of one attack filling the cap
HUMAN_EXCLUDE_ATTACKS = {"paraphrase", "synonym"}  # rewritten humans would clash with humanised AI
MIN_CHARS = 200
MAX_CHARS = 12000
REGENERATE = True       # the first slice kept only unperturbed humans; set False after this rebuild

import csv
import hashlib
import io
import json
import os
import re
import socket
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

T0 = time.time()
OUT_DIR = Path("runs/v2")
OUT_DIR.mkdir(parents=True, exist_ok=True)
COMBINED = OUT_DIR / "raid_v2.jsonl"
META_PATH = OUT_DIR / "raid_v2_meta.json"
_WS = re.compile(r"\s+")


def stage(message):
    print(f"\n[raid +{time.time() - T0:5.0f}s] {message}", flush=True)


token = os.environ.get("HF_TOKEN") or ""
if not token:  # never getpass: it blocks a notebook cell
    try:
        from kaggle_secrets import UserSecretsClient

        token = UserSecretsClient().get_secret("HF_TOKEN") or ""
    except Exception as exc:
        print("[raid] Kaggle secret unavailable:", exc, flush=True)
if not token:
    try:
        from google.colab import userdata

        token = userdata.get("HF_TOKEN") or ""
    except Exception as exc:
        print("[raid] Colab secret unavailable:", exc, flush=True)
if not token:
    raise SystemExit(
        "[raid] HF_TOKEN missing. Kaggle: Add-ons > Secrets, add HF_TOKEN (write scope) and "
        "enable it for this notebook; Colab: key icon in the left sidebar. Then re-run."
    )
os.environ["HF_TOKEN"] = token
print("[raid] HF_TOKEN present (value not printed)", flush=True)

try:
    socket.create_connection(("huggingface.co", 443), timeout=10).close()
except OSError as exc:
    raise SystemExit("[raid] no internet: Kaggle Settings > Internet > On, then re-run.") from exc

from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

api = HfApi(token=token)


def norm(text):
    return _WS.sub(" ", text or "").strip()


def archive(rows):
    """Merge whatever domains are already on disk into one file (resume-friendly)."""
    existing = {}
    if COMBINED.is_file():
        for line in COMBINED.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec.get("text"), str) and rec.get("label") in (0, 1):
                existing[hashlib.sha1(rec["text"].encode("utf-8", "ignore")).hexdigest()] = rec
    for rec in rows:
        existing[hashlib.sha1(rec["text"].encode("utf-8", "ignore")).hexdigest()] = rec
    ordered = sorted(existing.values(), key=lambda r: (r.get("domain", ""), r.get("pair_id", "")))
    tmp = COMBINED.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for rec in ordered:
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp.replace(COMBINED)
    return len(ordered)


def push(files, message):
    for path, dest in files:
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=dest,
            repo_id=HUB_ID,
            commit_message=message,
        )
    print("[raid] pushed", ", ".join(dest for _, dest in files), flush=True)


def restore():
    if REGENERATE:
        for path in (COMBINED, META_PATH):
            if path.is_file():
                path.unlink()
        return
    try:
        path = hf_hub_download(repo_id=HUB_ID, filename="v2/raid_v2.jsonl", token=token)
        COMBINED.write_text(Path(path).read_text(encoding="utf-8"), encoding="utf-8")
        print("[raid] restored the pushed slice from the Hub", flush=True)
    except Exception as exc:
        print("[raid] no pushed slice to restore:", repr(exc), flush=True)
    try:
        path = hf_hub_download(repo_id=HUB_ID, filename="v2/raid_v2_meta.json", token=token)
        META_PATH.write_text(Path(path).read_text(encoding="utf-8"), encoding="utf-8")
    except Exception:
        pass


def load_meta():
    if META_PATH.is_file():
        try:
            return json.loads(META_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {"windows": {k: list(v) for k, v in WINDOWS.items()}, "domains": {}, "completed": []}


def stream_domain(domain, start_gb, end_gb):
    """Yield 11-column rows from a byte range of train.csv, filtering by domain."""
    start = int(start_gb * 1e9)
    end = int(end_gb * 1e9)
    request = urllib.request.Request(RAID_URL, headers={"Range": f"bytes={start}-{end - 1}"})
    response = urllib.request.urlopen(request, timeout=600)
    if response.status != 206:
        response.close()
        raise RuntimeError(f"range request returned {response.status}; refusing to stream the full CSV")
    wrapper = io.TextIOWrapper(response, encoding="utf-8", errors="ignore", newline="")
    reader = csv.reader(wrapper)
    skipped = False
    try:
        for row in reader:
            if not skipped:  # the range starts mid-row
                skipped = True
                continue
            if len(row) != 11:
                continue
            if row[7].strip().lower() != domain:
                continue
            yield row
    except csv.Error:
        pass
    finally:
        response.close()


def to_record(row, domain):
    model = row[3].strip().lower()
    attack = row[6].strip().lower()
    if model == "human":
        if attack in HUMAN_EXCLUDE_ATTACKS:
            return None
        label, variant = 0, "raw"  # attacked human text is still human-labelled in RAID
    elif attack == "none":
        label, variant = 1, "raw"
    elif attack in ("paraphrase", "synonym"):
        label, variant = 1, "humanized"
    else:
        return None
    text = norm(row[10])
    if not (MIN_CHARS <= len(text) <= MAX_CHARS):
        return None
    prompt = norm(row[9]) or norm(row[8])
    group = row[2].strip() or row[0].strip()
    return {
        "text": text,
        "label": label,
        "generator": "human" if label == 0 else model,
        "source": "raid",
        "prompt": prompt[:500],
        "domain": domain,
        "variant": variant,
        "attack": attack,
        "group": group,
        "pair_id": group,
    }


def build_domain(domain, start_gb, end_gb):
    counts = {"human": 0, "ai": 0, "humanized": 0}
    models = Counter()
    human_attacks = Counter()
    per_model = defaultdict(int)
    per_attack = defaultdict(int)
    rows = 0
    stage(f"{domain}: reading {start_gb:.2f}-{end_gb:.2f} GB")
    for row in stream_domain(domain, start_gb, end_gb):
        rows += 1
        rec = to_record(row, domain)
        if rec is None:
            continue
        if rec["label"] == 0:
            if counts["human"] >= HUMAN_CAP or per_attack[rec["attack"]] >= HUMAN_ATTACK_CAP:
                continue
            per_attack[rec["attack"]] += 1
            human_attacks[rec["attack"]] += 1
            counts["human"] += 1
        elif rec["variant"] == "humanized":
            if counts["humanized"] >= HUMANIZED_CAP:
                continue
            if per_model[rec["generator"]] >= HUMANIZED_MODEL_CAP:
                continue
            counts["humanized"] += 1
        else:
            if counts["ai"] >= AI_CAP or per_model[rec["generator"]] >= MODEL_CAP:
                continue
            per_model[rec["generator"]] += 1
            counts["ai"] += 1
        models[rec["generator"]] += 1
        yield rec
        if rows % 50000 == 0:
            print(
                f"[raid] {domain}: {rows} rows | human {counts['human']}"
                f" ai {counts['ai']} humanized {counts['humanized']} | {time.time() - T0:.0f}s",
                flush=True,
            )
        if (
            counts["human"] >= HUMAN_CAP
            and counts["ai"] >= AI_CAP
            and counts["humanized"] >= HUMANIZED_CAP
        ):
            print(f"[raid] {domain}: caps reached after {rows} rows", flush=True)
            break
    stage(
        f"{domain}: {rows} rows -> human {counts['human']}, ai {counts['ai']},"
        f" humanized {counts['humanized']}; models {dict(models.most_common(6))}"
        f"; human attacks {dict(human_attacks.most_common(6))}"
    )


def length_summary():
    stats = defaultdict(list)
    for line in COMBINED.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        stats[(rec["domain"], "human" if rec["label"] == 0 else "ai")].append(len(rec["text"]))
    summary = {}
    for (domain, cls), values in sorted(stats.items()):
        values.sort()
        n = len(values)
        summary[f"{domain}/{cls}"] = {
            "n": n,
            "p10": values[int(n * 0.10)],
            "p50": values[int(n * 0.50)],
            "p90": values[min(n - 1, int(n * 0.90))],
        }
    return summary


restore()
meta = load_meta()
completed = set(meta.get("completed", []))
print("[raid] completed domains:", sorted(completed), flush=True)

for domain, (start_gb, end_gb) in WINDOWS.items():
    if domain in completed:
        print(f"[raid] {domain}: already in the pushed slice, skipping", flush=True)
        continue
    rows = list(build_domain(domain, start_gb, end_gb))
    if not rows:  # a wrong byte window must not be recorded as done
        print(
            f"[raid] {domain}: 0 rows in {start_gb:.2f}-{end_gb:.2f} GB; window is wrong,"
            " not marking it complete",
            flush=True,
        )
        continue
    total = archive(rows)
    meta["domains"][domain] = {
        "rows": len(rows),
        "human": sum(1 for r in rows if r["label"] == 0),
        "ai": sum(1 for r in rows if r["label"] == 1 and r["variant"] == "raw"),
        "humanized": sum(1 for r in rows if r["variant"] == "humanized"),
        "human_attacks": dict(Counter(r["attack"] for r in rows if r["label"] == 0)),
    }
    meta["completed"] = sorted(completed | {domain})
    meta["updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    completed.add(domain)
    print(f"[raid] combined slice now {total} rows", flush=True)
    push(
        [(COMBINED, "v2/raid_v2.jsonl"), (META_PATH, "v2/raid_v2_meta.json")],
        f"gradientTrace RAID slice: {domain} done ({len(rows)} rows)",
    )

meta["lengths"] = length_summary()
meta["combined_rows"] = sum(1 for _ in COMBINED.open("r", encoding="utf-8"))
META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
push([(META_PATH, "v2/raid_v2_meta.json")], "gradientTrace RAID slice: length summary")

print(f"\n[raid] domains: {meta['domains']}", flush=True)
print(f"[raid] lengths: {json.dumps(meta['lengths'], indent=2)}", flush=True)
print(f"[raid] done in {time.time() - T0:.0f}s -> {HUB_ID}/v2/raid_v2.jsonl", flush=True)
