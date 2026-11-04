"""Fit the shipped fusion on held-out slices only, and save it as a deployable artifact.

`colab/hybrid_eval.py` fits on the gate itself, which makes any gate submission of that artifact
an in-bank measurement (see AGENTS.md §7.9). This script fits the same feature set on the bank
cache's rows — which come from the pilot's own held-out slices and never saw a gate row — so the
saved weights can go through `colab/external_gate.py` as a genuine acceptance submission.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from gradienttrace.hybrid import (  # noqa: E402
    DETECTOR_NAMES,
    STYLO_NAMES,
    build_static,
    oof_fusion,
    save_fusion,
    stratify,
)

BANK_CACHE = os.path.join(ROOT, "hybrid_bank_cache.npz")
OUT_DIR = os.environ.get("OUT_DIR", os.path.join(ROOT, "hybrid_fusion_external"))
SHIPPED = DETECTOR_NAMES + STYLO_NAMES


def operating(score, y, target_fpr):
    thr = float(np.quantile(score[y == 0], 1.0 - target_fpr))
    return float((score[y == 1] >= thr).mean()), thr


def main():
    with np.load(BANK_CACHE, allow_pickle=True) as data:
        texts_b = list(data["texts"])
        probs_b, emb_b = data["probs"], data["embeddings"]
        lab_b, grp_b, dom_b = data["labels"], data["groups"], data["domain"]
    X_bank = build_static(probs_b, texts_b)
    oof_bank, coefs, fusion, _ = oof_fusion(SHIPPED, X_bank, emb_b, lab_b, grp_b, stratify(lab_b, dom_b))
    _, thr = operating(oof_bank, lab_b, 0.01)
    fusion.threshold = thr
    save_fusion(OUT_DIR, fusion, emb_b, lab_b, grp_b, "Yashhh999/gradientTrace-v2-pilot", 8)
    print("[fit] shipped fusion fit on %d held-out-slice rows (features=%d)" % (len(lab_b), len(fusion.names)))
    print("[fit] bank-side 1%%-FPR operating threshold %.4f; bank TPR@1%% %.4f" % (thr, operating(oof_bank, lab_b, 0.01)[0]))
    sign = (np.sign(coefs) == np.sign(coefs.mean(axis=0))).mean(axis=0)
    print("[fit] sign agreement: min %.2f over %d features" % (sign.min(), len(sign)))
    print("[saved] %s/fusion.json + bank.npz" % OUT_DIR)


if __name__ == "__main__":
    main()
