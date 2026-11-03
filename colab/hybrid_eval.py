"""Hybrid acceptance run: gradientTrace detector + a reference LM, fused and scored on the
frozen external gate, then transferred to extra texts outside the gate's registers.

The fusion is fit with 5-fold out-of-fold CV, so a row never contributes to the weights that
score it. Each system is then compared at its own operating point, not at a shared threshold,
because the fused score is not calibrated to 0.5 the way the detector is.

What this measures: whether a second, independent signal moves the gate ROC, and whether the
fused score keeps its calibration when applied to a different register. It does not establish
generalisation to unseen generators: the gate's AI side is one model (Qwen2.5-3B).

Run on any CUDA box; ~2 min for the 1,600-row gate with a 124M reference model.
"""

import json
import math
import os
import re
import sys

DET_ID = "Yashhh999/gradientTrace-v2-pilot"   # detector under test
REF_ID = "openai-community/gpt2"              # reference LM; must be unseen in training
GATE_REPO = "Yashhh999/gradientTrace"
GATE_NAME = "external_gate_v1.jsonl"
PROBE_TXT = os.path.expanduser("~/texts20.txt")  # "" skips the transfer stage
BURST_WINDOW = 8
REF_MAX_TOKENS = 1024
BATCH = 16

NAMES = ["det_logit", "bpb", "nll_std", "burst", "tail", "log_chars"]
SETS = {
    "gpt2-only": [1, 2, 3, 4, 5],
    "hybrid det+bpb": [0, 1],
    "hybrid all": [0, 1, 2, 3, 4, 5],
    "hybrid no-length": [0, 1, 2, 3, 4],
}


def repo_dir():
    for cand in ("/kaggle/working/gradientTrace", "/content/gradientTrace", os.path.expanduser("~/gradientTrace")):
        if os.path.isdir(cand):
            return cand
    return os.getcwd()


sys.path.insert(0, repo_dir())

import numpy as np  # noqa: E402
import torch  # noqa: E402
from huggingface_hub import hf_hub_download, snapshot_download  # noqa: E402

from gradienttrace.data import Sample  # noqa: E402
from gradienttrace.evaluate import load_model_and_tokenizer, predict_probs  # noqa: E402
from gradienttrace.metrics import binary_detection_metrics  # noqa: E402


def gate_records():
    path = hf_hub_download(GATE_REPO, GATE_NAME, repo_type="model")
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def probe_rows(path):
    """Chat-UI explainer blocks: '**N. Title' on its own line, then the body."""
    if not path or not os.path.isfile(path):
        return []
    raw = open(path, encoding="utf-8").read()
    rows = []
    for block in re.split(r"(?m)^(?=\*\*\d+\.)", raw):
        m = re.match(r"\*\*(\d+)\.\s*(.*?)\*\*", block.strip())
        if m:
            rows.append(("explainer-%s" % m.group(1), " ".join(block.strip()[m.end():].split())))
    return rows


@torch.no_grad()
def reference_features(texts, model, tok, device):
    """Per-text surprisal stats from the reference LM, teacher-forced over its own tokens."""
    stats = []
    for text in texts:
        ids = tok(text, add_special_tokens=False, truncation=True, max_length=REF_MAX_TOKENS)["input_ids"]
        if len(ids) < 3:
            stats.append((0.0, 0.0, 0.0, 0.0))
            continue
        x = torch.tensor([ids], device=device)
        logits = model(input_ids=x).logits[0, :-1]
        nll = (-torch.log_softmax(logits.float(), dim=-1).gather(1, x[0, 1:, None])[:, 0]).cpu().numpy()
        n_bytes = max(len(tok.decode(ids[1:]).encode("utf-8")), 1)
        bpb = float(nll.sum() / (math.log(2.0) * n_bytes))
        k = max(1, int(round(0.2 * len(nll))))
        tail = float(np.sort(nll)[-k:].mean() - nll.mean())
        wins = len(nll) // BURST_WINDOW
        burst = float(nll[: wins * BURST_WINDOW].reshape(wins, BURST_WINDOW).mean(axis=1).std()) if wins >= 2 else 0.0
        stats.append((bpb, float(nll.std()), burst, tail))
    return stats


def build_features(det_p, ref, chars):
    p = np.clip(np.asarray(det_p, dtype=np.float64).ravel(), 1e-6, 1 - 1e-6)
    return np.column_stack(
        [
            np.log(p / (1.0 - p)),
            np.asarray(ref, dtype=np.float64),
            np.log1p(np.asarray(chars, dtype=np.float64)),
        ]
    )


def cv_fit(X, y, strat, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    probs = np.zeros(len(y))
    coefs = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, strat):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=1.0, max_iter=4000).fit(sc.transform(X[tr]), y[tr])
        probs[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
        coefs.append(clf.coef_[0])
    scaler = StandardScaler().fit(X)
    final = LogisticRegression(C=1.0, max_iter=4000).fit(scaler.transform(X), y)
    return probs, np.vstack(coefs), scaler, final


def fmt(tag, score, y, mask=None):
    idx = slice(None) if mask is None else mask
    m = binary_detection_metrics(y[idx], score[idx])
    fpr05 = float((score[idx][y[idx] == 0] >= 0.5).mean())
    return "%-22s AUROC %.4f  TPR@1%% %.4f  TPR@5%% %.4f  FPR@0.5 %.4f" % (
        tag, m["auroc"], m["tpr_at_fpr_0.01"], m["tpr_at_fpr_0.05"], fpr05
    )


def operating(score, y, target_fpr):
    thr = float(np.quantile(score[y == 0], 1.0 - target_fpr))
    return float((score[y == 1] >= thr).mean()), thr


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    recs = gate_records()
    texts = [r["text"] for r in recs]
    y = np.array([int(r["label"]) for r in recs])
    genre = np.array([r["genre"] for r in recs])
    rows = probe_rows(PROBE_TXT)
    print("[hybrid] gate %d rows | probe texts %d | detector %s | reference %s"
          % (len(recs), len(rows), DET_ID, REF_ID), flush=True)

    det_dir = snapshot_download(
        DET_ID,
        allow_patterns=["config.json", "model.safetensors", "tokenizer*", "special_tokens_map.json", "spm.model"],
    )
    model, tok = load_model_and_tokenizer(det_dir)
    model.to(device).eval()
    gate_det = np.asarray(
        predict_probs(model, tok, [Sample(text=t, label=1, generator="x", source="gate") for t in texts],
                      device, batch_size=BATCH, desc="hybrid-detector"),
        dtype=np.float64,
    )
    probe_det = np.asarray(
        predict_probs(model, tok, [Sample(text=t, label=1, generator="x", source="probe") for _, t in rows],
                      device, batch_size=BATCH, desc="hybrid-probe-detector"),
        dtype=np.float64,
    ) if rows else np.zeros(0)
    del model
    torch.cuda.empty_cache()

    ref_model = AutoModelForCausalLM.from_pretrained(REF_ID).to(device).eval()
    ref_tok = AutoTokenizer.from_pretrained(REF_ID)
    gate_ref = [s for t in texts for s in reference_features([t], ref_model, ref_tok, device)]
    probe_ref = [s for _, t in rows for s in reference_features([t], ref_model, ref_tok, device)] if rows else []
    del ref_model
    torch.cuda.empty_cache()
    print("[hybrid] features done", flush=True)

    X = build_features(gate_det, gate_ref, [len(t) for t in texts])
    Xp = build_features(probe_det, probe_ref, [len(t) for _, t in rows]) if rows else None
    strat = np.char.add(np.char.add(y.astype(str), "_"), genre)
    news, revs = genre == "news", genre == "reviews"

    print("\n=== frozen gate, 5-fold out-of-fold CV ===")
    print(fmt("detector", gate_det, y))
    print("  " + fmt("detector/news", gate_det, y, news))
    print("  " + fmt("detector/reviews", gate_det, y, revs))
    for name, cols in SETS.items():
        probs, coefs, _, _ = cv_fit(X[:, cols], y, strat)
        print(fmt(name, probs, y))
        print("  " + fmt(name + "/news", probs, y, news))
        print("  " + fmt(name + "/reviews", probs, y, revs))
        print("  coefs " + "  ".join("%s %+.2f" % (NAMES[c], coefs.mean(axis=0)[c]) for c in range(len(cols))))

    np.savez_compressed(os.path.join(os.getcwd(), "hybrid_gate_features.npz"), X=X, y=y, genre=genre, det_p=gate_det)
    if Xp is None:
        print("\n[hybrid] no PROBE_TXT; transfer stage skipped")
        return

    print("\n=== transfer to %d texts outside the gate's registers ===" % len(rows))
    for name in ("hybrid det+bpb", "hybrid no-length", "hybrid all"):
        cols = SETS[name]
        oof, _, scaler, clf = cv_fit(X[:, cols], y, strat)
        s_probe = clf.predict_proba(scaler.transform(Xp[:, cols]))[:, 1]
        d1, dt1 = operating(gate_det, y, 0.01)
        h1, ht1 = operating(oof, y, 0.01)
        d5, dt5 = operating(gate_det, y, 0.05)
        h5, ht5 = operating(oof, y, 0.05)
        print("\n%s" % name)
        print("  own operating points: detector TPR@1%% %.4f (thr %.4f) TPR@5%% %.4f (thr %.4f)" % (d1, dt1, d5, dt5))
        print("                        hybrid  TPR@1%% %.4f (thr %.4f) TPR@5%% %.4f (thr %.4f)" % (h1, ht1, h5, ht5))
        print("  flagged above own 1%%-FPR threshold: detector %d/%d, hybrid %d/%d"
              % (int((probe_det >= dt1).sum()), len(rows), int((s_probe >= ht1).sum()), len(rows)))
        print("  flagged above 0.5: detector %d/%d, hybrid %d/%d"
              % (int((probe_det >= 0.5).sum()), len(rows), int((s_probe >= 0.5).sum()), len(rows)))
        print("  %-16s %6s %8s %8s %7s %7s" % ("text", "chars", "det_p", "hyb_p", "bpb", "burst"))
        for i, (key, text) in enumerate(rows):
            print("  %-16s %6d %8.4f %8.4f %7.3f %7.3f" % (key, len(text), probe_det[i], s_probe[i], Xp[i, 1], Xp[i, 3]))


if __name__ == "__main__":
    main()
