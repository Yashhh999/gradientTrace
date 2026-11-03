"""Hybrid acceptance run: the gradientTrace detector plus stylometry and embedding geometry,
fused by logistic regression and scored on the frozen external gate.

There is no second language model here. The extra signals are computed from the detector itself:

  * `stylometry` — 33 hand-counted surface statistics per text;
  * `vector_features` — k-NN and centroid geometry in the detector's own pooled representation,
    measured against labelled gate rows. The scalar logit throws that geometry away.

Vector features are rebuilt **inside every CV fold** from training-fold references only, and a
row's pair partner is always blocked, so the reported gain is not leakage. Each system is then
compared at its own FPR-calibrated operating point, because the fused score is not calibrated
to 0.5 the way the detector is.

Run on any box with the requirements installed; CPU works (the gate is 1,600 short texts).
"""

import json
import os
import re
import sys

DET_ID = os.environ.get("DET_ID", "Yashhh999/gradientTrace-v2-pilot")  # detector under test
GATE_REPO = "Yashhh999/gradientTrace"
GATE_NAME = "external_gate_v1.jsonl"
PROBE_TXT = os.environ.get("PROBE_TXT", os.path.expanduser("~/texts20.txt"))  # "" skips the transfer stage
OUT_DIR = os.environ.get("OUT_DIR", "hybrid_fusion")  # fusion.json + bank.npz, loadable by HybridScorer
K_NEIGHBOURS = 8
BATCH = 16
SHIP_SET = os.environ.get("SHIP_SET", "det + stylometry")  # the set that survived the external-bank test


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
from gradienttrace.evaluate import load_model_and_tokenizer, predict_probs_and_embeddings  # noqa: E402
from gradienttrace.hybrid import (  # noqa: E402
    DETECTOR_NAMES,
    FEATURES,
    STYLO_NAMES,
    VECTOR_NAMES,
    HybridScorer,
    build_static,
    oof_fusion,
    save_fusion,
    stratify,
)
from gradienttrace.metrics import binary_detection_metrics  # noqa: E402

SETS = {
    "stylometry only": STYLO_NAMES,
    "vectors only": VECTOR_NAMES,
    "det + stylometry": DETECTOR_NAMES + STYLO_NAMES,
    "det + vectors": DETECTOR_NAMES + VECTOR_NAMES,
    "hybrid all": FEATURES,
}


def gate_records():
    path = hf_hub_download(GATE_REPO, GATE_NAME, repo_type="model")
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def probe_rows(path):
    """Chat-UI explainer blocks: '**N. Title' on its own line, then the body."""
    if not path or not os.path.isfile(path):
        return []
    rows = []
    for block in re.split(r"(?m)^(?=\*\*\d+\.)", open(path, encoding="utf-8").read()):
        m = re.match(r"\*\*(\d+)\.\s*(.*?)\*\*", block.strip())
        if m:
            rows.append(("explainer-%s" % m.group(1), " ".join(block.strip()[m.end():].split())))
    return rows


def fmt(tag, score, y, mask=None):
    idx = slice(None) if mask is None else mask
    m = binary_detection_metrics(y[idx], score[idx])
    fpr05 = float((score[idx][y[idx] == 0] >= 0.5).mean())
    return "%-20s AUROC %.4f  TPR@1%% %.4f  TPR@5%% %.4f  FPR@0.5 %.4f" % (
        tag, m["auroc"], m["tpr_at_fpr_0.01"], m["tpr_at_fpr_0.05"], fpr05
    )


def operating(score, y, target_fpr):
    thr = float(np.quantile(score[y == 0], 1.0 - target_fpr))
    return float((score[y == 1] >= thr).mean()), thr


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    recs = gate_records()
    texts = [r["text"] for r in recs]
    y = np.asarray([int(r["label"]) for r in recs], dtype=np.int64)
    genre = np.asarray([r["genre"] for r in recs])
    group_ids = {}
    groups = np.asarray(
        [group_ids.setdefault(str(r.get("pair_id") or r.get("id") or i), len(group_ids)) for i, r in enumerate(recs)],
        dtype=np.int64,
    )
    rows = probe_rows(PROBE_TXT)
    print("[hybrid] gate %d rows, %d groups | probe %d | detector %s | device %s"
          % (len(recs), len(group_ids), len(rows), DET_ID, device), flush=True)

    det_dir = DET_ID if os.path.isdir(DET_ID) else snapshot_download(
        DET_ID,
        allow_patterns=["config.json", "model.safetensors", "tokenizer*", "special_tokens_map.json", "spm.model"],
    )
    model, tok = load_model_and_tokenizer(det_dir)
    model.to(device).eval()
    probs, embeddings = predict_probs_and_embeddings(
        model, tok, [Sample(text=t, label=1, generator="x", source="gate") for t in texts],
        device, batch_size=BATCH, desc="hybrid-gate",
    )
    print("[hybrid] detector done: %d x %d embeddings" % embeddings.shape, flush=True)

    det_p = probs.astype(np.float64)
    p = np.clip(det_p, 1e-6, 1 - 1e-6)
    X_static = build_static(det_p, texts)
    strat = stratify(y, genre)
    news, revs = genre == "news", genre == "reviews"

    print("\n=== frozen gate, 5-fold out-of-fold, fold-safe k-NN ===")
    print(fmt("detector", det_p, y))
    print("  " + fmt("detector/news", det_p, y, news))
    print("  " + fmt("detector/reviews", det_p, y, revs))
    fits = {}
    for name, cols in SETS.items():
        oof, coefs, fusion, _ = oof_fusion(cols, X_static, embeddings.astype(np.float64), y, groups, strat, k=K_NEIGHBOURS)
        fits[name] = (oof, fusion)
        print(fmt(name, oof, y))
        print("  " + fmt(name + "/news", oof, y, news))
        print("  " + fmt(name + "/reviews", oof, y, revs))
        order = np.argsort(-np.abs(coefs.mean(axis=0)))[:8]
        mean = coefs.mean(axis=0)
        agree = (np.sign(coefs) == np.sign(mean)).mean(axis=0)
        print("  top coefs " + "  ".join(
            "%s %+.2f(%d%%)" % (cols[i], mean[i], round(100 * agree[i])) for i in order
        ))

    tpr1, thr1 = operating(det_p, y, 0.01)
    ship = SHIP_SET
    ship_oof, ship_fusion = fits[ship]
    _, ship_thr = operating(ship_oof, y, 0.01)
    ship_fusion.threshold = ship_thr
    save_fusion(OUT_DIR, ship_fusion, embeddings, y, groups, DET_ID, K_NEIGHBOURS)
    print("\n[saved] %s/fusion.json + bank.npz (%d bank rows) — ship set '%s'; gate 1%%-FPR cut: detector %.4f, fusion %.4f"
          % (OUT_DIR, len(y), ship, thr1, ship_thr))

    if not rows:
        print("[hybrid] no PROBE_TXT; transfer stage skipped")
        return

    print("\n=== transfer to %d texts outside the gate's registers ===" % len(rows))
    scorer = HybridScorer(det_dir, OUT_DIR, device=device, batch_size=BATCH, detector=(model, tok))
    probe_texts = [t for _, t in rows]
    full, det_probe, made = scorer.features(probe_texts)
    fused = ship_fusion.predict(full[:, [made.index(n) for n in ship_fusion.names]])
    marker_col = made.index("llm_marker_rate")
    knn_col = made.index("knn_margin") if "knn_margin" in made else None
    print("  operating points — detector TPR@1%% %.4f (thr %.4f) | ship fusion TPR@1%% %.4f (thr %.4f)"
          % (tpr1, thr1, operating(ship_oof, y, 0.01)[0], ship_thr))
    print("  flagged above own 1%%-FPR threshold: detector %d/%d, fusion %d/%d"
          % (int((det_probe >= thr1).sum()), len(rows), int((fused >= ship_thr).sum()), len(rows)))
    print("  flagged above 0.5: detector %d/%d, fusion %d/%d"
          % (int((det_probe >= 0.5).sum()), len(rows), int((fused >= 0.5).sum()), len(rows)))
    head = "  %-14s %6s %8s %8s %9s" % ("text", "chars", "det_p", "hyb_p", "llm_rate")
    print(head + ("" if knn_col is None else " %9s" % "knn_marg"))
    for i, (key, text) in enumerate(rows):
        line = "  %-14s %6d %8.4f %8.4f %9.4f" % (key, len(text), det_probe[i], fused[i], full[i, marker_col])
        print(line + ("" if knn_col is None else " %9.4f" % full[i, knn_col]))


if __name__ == "__main__":
    main()
