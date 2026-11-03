"""Score public AI-text detectors on the frozen external gate.

Same 1,600 rows, same unseen-Qwen AI side, same human-calibrated 1% FPR headline metric the
gradientTrace models are judged on, so the comparison is apples to apples. Every checkpoint's
label convention is read from its own `config.id2label` and printed: this project has already been
burned once by a source whose 1 meant human (MAGE), so no model is trusted to follow the
"1 = AI" convention used here.

Usage: python colab/gate_baselines.py [--only <substr>] [--max-rows N]
"""

import argparse
import json
import os
import re
import sys
import time

GATE_REPO = "Yashhh999/gradientTrace"
GATE_NAME = "external_gate_v1.jsonl"
BATCH = 16
MAX_LEN = 512

CANDIDATES = [
    "openai-community/roberta-large-openai-detector",
    "openai-community/roberta-base-openai-detector",
    "fakespot-ai/roberta-base-ai-text-detection-v1",
    "tabularisai/ai-text-detection",
    "shunya510hi/gpt4-text-detector",
]

AI_WORDS = ("ai", "gpt", "machine", "generated", "fake", "bot", "synthetic", "model", "not-human", "non-human")
HUMAN_WORDS = ("human", "real", "true", "authentic", "natural", "original")


def repo_dir():
    for cand in ("/kaggle/working/gradientTrace", "/content/gradientTrace", os.path.expanduser("~/gradientTrace")):
        if os.path.isdir(cand):
            return cand
    return os.getcwd()


sys.path.insert(0, repo_dir())

import numpy as np  # noqa: E402
import torch  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402

from gradienttrace.metrics import binary_detection_metrics  # noqa: E402


def gate_rows():
    path = hf_hub_download(GATE_REPO, GATE_NAME, repo_type="model")
    with open(path, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    return rows


# Some checkpoints ship no usable label names ("label_0"/"label_1" or nothing at all). Where the
# convention is documented by a sibling checkpoint from the same training run, it is recorded here
# with its source; anything left unresolved is scored in both directions rather than guessed.
LABEL_OVERRIDES = {
    # openai-community/roberta-base-openai-detector declares {0: fake, 1: real}; the large
    # checkpoint comes from the same release and ships no id2label.
    "openai-community/roberta-large-openai-detector": 0,
}


def ai_label_index(config, model_id):
    """Which output index means machine text, read from the checkpoint or its documented sibling."""
    id2label = {int(k): str(v).lower() for k, v in (config.id2label or {}).items()}
    if model_id in LABEL_OVERRIDES:
        return LABEL_OVERRIDES[model_id], id2label or "override"
    ai = [i for i, name in id2label.items() if any(w in name for w in AI_WORDS)]
    if len(id2label) != 2:
        raise ValueError("expected a binary detector, got %s" % id2label)
    if len(ai) != 1:
        raise ValueError("cannot resolve the AI class from %s" % id2label)
    return ai[0], id2label


@torch.inference_mode()
def score(model, tok, texts, ai_index, device, batch=BATCH):
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    probs = np.zeros(len(texts), dtype=np.float64)
    for start in range(0, len(texts), batch):
        idx = order[start : start + batch]
        enc = tok([texts[i] for i in idx], truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt").to(device)
        out = model(**enc)
        p = torch.softmax(out.logits.float(), dim=-1)[:, ai_index].cpu().numpy()
        probs[idx] = p
    return probs


def report(tag, probs, y, genre):
    news, revs = genre == "news", genre == "reviews"
    rows = {}
    for name, mask in (("pooled", np.ones(len(y), bool)), ("news", news), ("reviews", revs)):
        m = binary_detection_metrics(y[mask], probs[mask])
        rows[name] = {
            "auroc": round(m["auroc"], 4),
            "tpr_at_1pct_fpr": round(m["tpr_at_fpr_0.01"], 4),
            "tpr_at_5pct_fpr": round(m["tpr_at_fpr_0.05"], 4),
            "fpr_at_0_5": round(float((probs[mask][y[mask] == 0] >= 0.5).mean()), 4),
            "mean_human": round(float(probs[mask][y[mask] == 0].mean()), 4),
            "mean_ai": round(float(probs[mask][y[mask] == 1].mean()), 4),
        }
    print("%-42s AUROC %.4f  TPR@1%% %.4f  TPR@5%% %.4f  FPR@0.5 %.4f  meanH %.4f  meanAI %.4f"
          % (tag, rows["pooled"]["auroc"], rows["pooled"]["tpr_at_1pct_fpr"], rows["pooled"]["tpr_at_5pct_fpr"],
             rows["pooled"]["fpr_at_0_5"], rows["pooled"]["mean_human"], rows["pooled"]["mean_ai"]))
    for name in ("news", "reviews"):
        r = rows[name]
        print("    %-8s AUROC %.4f  TPR@1%% %.4f  TPR@5%% %.4f  FPR@0.5 %.4f  meanH %.4f  meanAI %.4f"
              % (name, r["auroc"], r["tpr_at_1pct_fpr"], r["tpr_at_5pct_fpr"], r["fpr_at_0_5"], r["mean_human"], r["mean_ai"]))
    return rows


def main():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="substring filter on model ids")
    ap.add_argument("--max-rows", type=int, default=0, help="debug: score only the first N rows")
    args = ap.parse_args()

    device = torch.device("cpu")
    torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))
    rows = gate_rows()
    texts = [r["text"] for r in rows]
    y = np.asarray([int(r["label"]) for r in rows], dtype=np.int64)
    genre = np.asarray([r["genre"] for r in rows])
    if args.max_rows:
        texts, y, genre = texts[: args.max_rows], y[: args.max_rows], genre[: args.max_rows]
    print("[baselines] gate rows %d (label 1 = AI), device %s, threads %d" % (len(texts), device, torch.get_num_threads()), flush=True)

    out = {}
    for mid in CANDIDATES:
        if args.only and args.only not in mid:
            continue
        started = time.monotonic()
        try:
            tok = AutoTokenizer.from_pretrained(mid)
            model = AutoModelForSequenceClassification.from_pretrained(mid).to(device).eval()
        except Exception as exc:
            print("%-42s FAILED to load %s: %s" % (mid, type(exc).__name__, str(exc)[:160]), flush=True)
            out[mid] = {"error": "%s: %s" % (type(exc).__name__, str(exc)[:200])}
            continue
        try:
            ai_index, id2label = ai_label_index(model.config, mid)
        except ValueError:
            # Unresolved convention: score both directions. AUROC flips to 1-a under the wrong
            # assignment, so a clear >0.5 identifies the AI class without assuming it.
            p0 = score(model, tok, texts, 0, device)
            print("%-42s id2label unresolved, both directions:" % mid, flush=True)
            out[mid] = {"unresolved": True, "ai_index_0": report(mid + " [P(0)]", p0, y, genre),
                        "ai_index_1": report(mid + " [P(1)]", 1.0 - p0, y, genre),
                        "seconds": round(time.monotonic() - started)}
            del model
            continue
        probs = score(model, tok, texts, ai_index, device)
        print("%-42s id2label=%s AI-index=%s  %.0fs" % (mid, id2label, ai_index, time.monotonic() - started), flush=True)
        out[mid] = {"id2label": id2label, "ai_index": ai_index, "seconds": round(time.monotonic() - started), **report(mid, probs, y, genre)}
        del model
    path = os.path.join(os.getcwd(), "gate_baselines.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("\nwrote %s" % path)


if __name__ == "__main__":
    main()
