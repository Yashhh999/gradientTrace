"""Router and abstention on top of the shipped fusion, computed from cached scores only.

The external-bank run showed the fusion helps news and hurts reviews, so a single global fusion is
the wrong policy. This script learns, from observable features, which system costs less at the
operating threshold, and marks disagreements for human review instead of forcing a call.

Every input comes from the caches the bank experiment produced, so nothing here loads a model and
the run is seconds on CPU. Thresholds are the bank's own out-of-fold 1%-FPR points.
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402

from gradienttrace.hybrid import DETECTOR_NAMES, STYLO_NAMES, build_static, oof_fusion, stratify  # noqa: E402
from gradienttrace.metrics import binary_detection_metrics  # noqa: E402

BANK_CACHE = os.path.join(ROOT, "hybrid_bank_cache.npz")
GATE_CACHE = os.path.join(ROOT, "hybrid_gate_cache.npz")
GATE_REPO = "Yashhh999/gradientTrace"
SHIPPED = DETECTOR_NAMES + STYLO_NAMES
C_FP, C_FN = 10.0, 1.0  # a false accusation costs ten times a miss

# Four off-register texts for the abstention check: two human classics/wikipedia leads and two AI
# texts whose detector scores span the confident-wrong range.
PROBE = [
    ("human-declaration", 0, "We hold these truths to be self-evident, that all men are created equal, that they are endowed by their Creator with certain unalienable Rights, that among these are Life, Liberty and the pursuit of Happiness."),
    ("wiki-gbr", 0, "The Great Barrier Reef is the world's largest coral reef system, composed of over 2,900 individual reefs and 900 islands stretching for over 2,300 kilometres over an area of approximately 344,400 square kilometres. The reef is located in the Coral Sea, off the coast of Queensland, Australia, and can be seen from outer space."),
    ("ai-explainer", 1, "Sunlight vanishes entirely about a thousand meters below the ocean's surface, leaving creatures in the bathypelagic zone to manufacture their own light. Through a chemical reaction involving a light-emitting molecule called luciferin and an enzyme known as luciferase, marine organisms produce cold light with almost no heat loss."),
    ("ai-product", 1, "Meet the AuraDesk Pro, the all-in-one standing desk built for people who take their workspace seriously. A whisper-quiet dual motor lifts up to 120 kg in under three seconds, while the integrated cable tray keeps your setup clean."),
]


def operating(score, y, target_fpr):
    thr = float(np.quantile(score[y == 0], 1.0 - target_fpr))
    return float((score[y == 1] >= thr).mean()), thr


def fmt(tag, score, y, mask=None):
    idx = slice(None) if mask is None else mask
    m = binary_detection_metrics(y[idx], score[idx])
    return "%-20s AUROC %.4f  TPR@1%% %.4f  TPR@5%% %.4f" % (
        tag, m["auroc"], m["tpr_at_fpr_0.01"], m["tpr_at_fpr_0.05"]
    )


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    with np.load(BANK_CACHE, allow_pickle=True) as data:
        texts_b = list(data["texts"])
        probs_b, emb_b = data["probs"], data["embeddings"]
        lab_b, grp_b, dom_b = data["labels"], data["groups"], data["domain"]
    with np.load(GATE_CACHE) as data:
        probs_g = data["probs"]
    rows = [json.loads(l) for l in open(hf_hub_download(GATE_REPO, "external_gate_v1.jsonl", repo_type="model"), encoding="utf-8") if l.strip()]
    y_gate = np.asarray([int(r["label"]) for r in rows], dtype=np.int64)
    genre = np.asarray([r["genre"] for r in rows])
    texts_g = [r["text"] for r in rows]
    print("[routing] bank %d rows, gate %d rows" % (len(lab_b), len(y_gate)), flush=True)

    X_bank = build_static(probs_b, texts_b)
    oof_bank, _, fusion, _ = oof_fusion(SHIPPED, X_bank, emb_b, lab_b, grp_b, stratify(lab_b, dom_b))

    det_b = probs_b
    _, thr_det = operating(det_b, lab_b, 0.01)
    _, thr_fus = operating(oof_bank, lab_b, 0.01)
    print("[routing] bank thresholds — detector %.4f, fusion %.4f (1%% FPR on the bank)" % (thr_det, thr_fus), flush=True)

    det_cost = C_FP * ((det_b >= thr_det) & (lab_b == 0)) + C_FN * ((det_b < thr_det) & (lab_b == 1))
    fus_cost = C_FP * ((oof_bank >= thr_fus) & (lab_b == 0)) + C_FN * ((oof_bank < thr_fus) & (lab_b == 1))
    cheaper = fus_cost < det_cost
    tie = fus_cost == det_cost
    print("[routing] fusion cheaper on %d bank rows, detector cheaper on %d, tie on %d"
          % (int(cheaper.sum()), int((~cheaper & ~tie).sum()), int(tie.sum())), flush=True)

    keep = ~tie
    scaler = StandardScaler().fit(X_bank[keep])
    policy = LogisticRegression(max_iter=4000, class_weight="balanced").fit(
        scaler.transform(X_bank[keep]), cheaper[keep].astype(int)
    )

    X_gate = build_static(probs_g, texts_g)
    fused_g = fusion.predict(X_gate)
    gate_policy_p = policy.predict_proba(scaler.transform(X_gate))[:, 1]
    route = gate_policy_p >= 0.5
    routed_g = np.where(route, fused_g, probs_g)

    news, revs = genre == "news", genre == "reviews"
    print("\n=== gate as unseen population: policy systems (bank thresholds) ===")
    for name_, mask in (("pooled", None), ("news", news), ("reviews", revs)):
        print(fmt("detector" + ("/%s" % name_ if name_ != "pooled" else ""), probs_g, y_gate, mask))
        print(fmt("fusion" + ("/%s" % name_ if mask is not None else ""), fused_g, y_gate, mask))
        print(fmt("routed" + ("/%s" % name_ if mask is not None else ""), routed_g, y_gate, mask))
    det_fpr = float((probs_g[y_gate == 0] >= thr_det).mean())
    fus_fpr = float((fused_g[y_gate == 0] >= thr_fus).mean())
    rou_fpr = float((routed_g[y_gate == 0] >= thr_det).mean())
    print("  human FPR on the gate at bank thresholds — detector %.4f, fusion %.4f, routed %.4f" % (det_fpr, fus_fpr, rou_fpr))
    print("  route to fusion: news %d/%d, reviews %d/%d" % (int(route[news].sum()), int(news.sum()), int(route[revs].sum()), int(revs.sum())))  # noqa: E501

    disagree = (probs_g >= thr_det) != (fused_g >= thr_fus)
    scored = ~disagree
    print("\n--- abstention (send the disagreement rows to human review) ---")
    print("  abstained %d/%d gate rows (%.1f%%): news %d/%d, reviews %d/%d"
          % (int(disagree.sum()), len(y_gate), 100 * disagree.mean(),
             int(disagree[news].sum()), int(news.sum()), int(disagree[revs].sum()), int(revs.sum())))
    for tag, score, thr in (("detector", probs_g, thr_det), ("fusion", fused_g, thr_fus), ("routed", routed_g, thr_det)):
        tp = int(((score >= thr) & (y_gate == 1) & scored).sum())
        fp = int(((score >= thr) & (y_gate == 0) & scored).sum())
        print("  %-9s on scored rows only: catches %d/%d AI, flags %d/%d humans (abstained %d AI, %d human)"
              % (tag, tp, int((y_gate[scored] == 1).sum()), fp, int((y_gate[scored] == 0).sum()),
                 int((y_gate[disagree] == 1).sum()), int((y_gate[disagree] == 0).sum())))

    print("\n--- off-register probe (bank-calibrated thresholds; abstain = systems disagree) ---")
    probe_texts = [t for _, _, t in PROBE]
    probe_probs = np.asarray([0.9365, 0.9980, 0.0411, 0.5255])  # measured on this session's pilot
    X_probe = build_static(probe_probs, probe_texts)
    fused_probe = fusion.predict(X_probe)
    policy_probe = policy.predict_proba(scaler.transform(X_probe))[:, 1]
    routed_probe = np.where(policy_probe >= 0.5, fused_probe, probe_probs)
    probe_disagree = (probe_probs >= thr_det) != (fused_probe >= thr_fus)
    for i, (key, label, _) in enumerate(PROBE):
        print("  %-14s %s det %.4f fusion %.4f routed %.4f P(route)=%.2f%s"
              % (key, "AI  " if label else "HUM ", probe_probs[i], fused_probe[i], routed_probe[i],
                 policy_probe[i], "  -> ABSTAIN" if probe_disagree[i] else ""))
    print("\n=== stylometry pruning (out-of-bank; bank-calibrated thresholds) ===")
    for tag, drop in (("full", []), ("-log_words", ["log_words"]), ("-sent_mean_words", ["sent_mean_words"]),
                      ("-llm_marker_rate", ["llm_marker_rate"]), ("-length-pair", ["log_words", "sent_mean_words"]),
                      ("-all three", ["log_words", "sent_mean_words", "llm_marker_rate"])):
        cols = [n for n in SHIPPED if n not in drop]
        oof_p, _, f_p, _ = oof_fusion(cols, X_bank, emb_b, lab_b, grp_b, stratify(lab_b, dom_b))
        _, thr_p = operating(oof_p, lab_b, 0.01)
        fused_p = f_p.predict(X_gate[:, [SHIPPED.index(n) for n in cols]])
        m_p = binary_detection_metrics(y_gate, fused_p)
        fpr_p = float((fused_p[y_gate == 0] >= thr_p).mean())
        caught_p = int((fused_p >= thr_p)[y_gate == 1].sum())
        print("  %-16s AUROC %.4f  TPR@1%% %.4f  bank-thr %.4f catches %d/800 (gate FPR %.4f)"
              % (tag, m_p["auroc"], m_p["tpr_at_fpr_0.01"], thr_p, caught_p, fpr_p))

    print("\n=== temperature scaling of the detector logit (fit on the bank) ===")
    from scipy.optimize import minimize_scalar

    logit_b = np.log(np.clip(probs_b, 1e-6, 1 - 1e-6) / (1 - np.clip(probs_b, 1e-6, 1 - 1e-6)))
    def nll(T):
        p = 1.0 / (1.0 + np.exp(-logit_b / T))
        return -float((lab_b * np.log(p + 1e-12) + (1 - lab_b) * np.log(1 - p + 1e-12)).mean())
    T = float(minimize_scalar(nll, bounds=(0.2, 5.0), method="bounded").x)
    logit_g = np.log(np.clip(probs_g, 1e-6, 1 - 1e-6) / (1 - np.clip(probs_g, 1e-6, 1 - 1e-6)))
    p_scaled = 1.0 / (1.0 + np.exp(-logit_g / T))
    m0 = binary_detection_metrics(y_gate, probs_g)
    print("  T=%.4f  detector FPR@0.5 unscaled %.4f -> scaled %.4f (AUROC %.4f either way)"
          % (T, float((probs_g[y_gate == 0] >= 0.5).mean()), float((p_scaled[y_gate == 0] >= 0.5).mean()), m0["auroc"]))
    print("  mean |scaled - unscaled| on the gate: %.4f" % float(np.abs(p_scaled - probs_g).mean()))


if __name__ == "__main__":
    main()
