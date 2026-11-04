"""Does the hybrid's gain survive when the k-NN bank is *not* the evaluation set?

`hybrid_eval.py` fits and evaluates on the frozen gate, so its bank is the population being
measured: out-of-fold stops any row seeing its own label and pair partners are blocked, but the
human/AI clusters are still the test set's own clusters. That makes those numbers an upper bound.

Here the bank comes from the model's own held-out eval slices (in-domain, unseen-generator,
humanized, held-out-domain poetry). The fusion is fit out-of-fold on that bank only, then applied
to the gate as a completely unseen population. If the gain is real, it survives the swap.

Inference-only: ~20 min of CPU for an 8k-row bank, no training and no GPU needed.
"""

import hashlib
import json
import os
import re
import sys

BANK_ROWS = 8000                     # balanced subsample of the held-out slices
K_NEIGHBOURS = 8
BATCH = 16
SEED = 0
CONFIG = "configs/pilot.yaml"        # the recipe the release pilot was trained on
DET_ID = "Yashhh999/gradientTrace-v2-pilot"
GATE_REPO = "Yashhh999/gradientTrace"
PROBE_TXT = os.environ.get("PROBE_TXT", "")  # explainer blocks; "" skips the transfer stage
POOLING = os.environ.get("POOLING", "cls")  # "cls" (head pooling) or "mean"
SUFFIX = "" if POOLING == "cls" else "-" + POOLING
CACHE = "hybrid_bank_cache%s.npz" % SUFFIX  # slice embeddings, pooling-specific
ROWS_CACHE = "hybrid_bank_rows.json"   # chosen bank rows, so the ~20 min data build runs once
GATE_CACHE = "hybrid_gate_cache%s.npz" % SUFFIX  # gate embeddings, so the gate is embedded once


def repo_dir():
    for cand in ("/kaggle/working/gradientTrace", "/content/gradientTrace", os.path.expanduser("~/gradientTrace")):
        if os.path.isdir(cand):
            return cand
    return os.getcwd()


ROOT = repo_dir()
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from huggingface_hub import hf_hub_download, snapshot_download  # noqa: E402

from gradienttrace.config import load_config  # noqa: E402
from gradienttrace.data import Sample, build_datasets  # noqa: E402
from gradienttrace.evaluate import load_model_and_tokenizer, predict_probs_and_embeddings  # noqa: E402
from gradienttrace.hybrid import (  # noqa: E402
    DETECTOR_NAMES,
    FEATURES,
    STYLO_NAMES,
    VECTOR_NAMES,
    assemble,
    build_static,
    oof_fusion,
    stratify,
    vector_features,
)
from gradienttrace.metrics import binary_detection_metrics  # noqa: E402

SETS = {
    "det + stylometry": DETECTOR_NAMES + STYLO_NAMES,
    "det + vectors": DETECTOR_NAMES + VECTOR_NAMES,
    "hybrid all": FEATURES,
}


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


def ensure_data_files():
    """The Kaggle/Colab cells download these; standalone runs need them too."""
    os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
    for local, name in (("data/raid_v2.jsonl", "v2/raid_v2.jsonl"), ("data/local.jsonl", "v2/news_pairs.jsonl")):
        path = os.path.join(ROOT, local)
        if os.path.isfile(path) and os.path.getsize(path) > 1_000_000:
            continue
        print("[bank] downloading %s -> %s" % (name, local), flush=True)
        src = hf_hub_download(GATE_REPO, name, repo_type="model")
        with open(src, "rb") as fh, open(path, "wb") as out:
            out.write(fh.read())


def group_id(sample):
    key = sample.group or (sample.prompt + "|" + sample.domain) or sample.text[:80]
    return int(hashlib.sha1(key.encode("utf-8", "ignore")).hexdigest()[:6], 16)


def pick_bank(bundle):
    """Balance the held-out slices by (slice, label, domain) and cap at BANK_ROWS rows."""
    pool = []
    for name in ("test_indomain", "test_ood_generators", "test_humanized", "test_heldout_domain"):
        for sample in bundle.slices.get(name, []):
            pool.append((name, sample))
    buckets = {}
    for name, sample in pool:
        buckets.setdefault((name, int(sample.label), sample.domain), []).append((name, sample))
    rng = np.random.default_rng(SEED)
    per = max(1, BANK_ROWS // max(1, len(buckets)))
    chosen = []
    for key in sorted(buckets):
        items = buckets[key]
        rng.shuffle(items)
        chosen.extend(items[:per])
    if len(chosen) > BANK_ROWS:
        rng.shuffle(chosen)
        chosen = chosen[:BANK_ROWS]
    counts = {}
    for name, sample in chosen:
        counts[(name, int(sample.label))] = counts.get((name, int(sample.label)), 0) + 1
    print("[bank] %d rows from %d buckets; by slice/label: %s"
          % (len(chosen), len(buckets), {" ".join(map(str, k)): v for k, v in sorted(counts.items())}), flush=True)
    return chosen


def embed(model, tok, device, texts, desc):
    samples = [Sample(text=t, label=1, generator="x", source="bank") for t in texts]
    probs, vectors = predict_probs_and_embeddings(model, tok, samples, device, batch_size=BATCH, desc=desc, pooling=POOLING)
    return probs.astype(np.float64), vectors.astype(np.float64)


def fmt(tag, score, y, mask=None):
    idx = slice(None) if mask is None else mask
    m = binary_detection_metrics(y[idx], score[idx])
    fpr05 = float((score[idx][y[idx] == 0] >= 0.5).mean())
    return "%-18s AUROC %.4f  TPR@1%% %.4f  TPR@5%% %.4f  FPR@0.5 %.4f" % (
        tag, m["auroc"], m["tpr_at_fpr_0.01"], m["tpr_at_fpr_0.05"], fpr05
    )


def operating(score, y, target_fpr):
    thr = float(np.quantile(score[y == 0], 1.0 - target_fpr))
    return float((score[y == 1] >= thr).mean()), thr


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensure_data_files()
    cfg = load_config(os.path.join(ROOT, CONFIG))

    recs = [json.loads(line) for line in open(hf_hub_download(GATE_REPO, "external_gate_v1.jsonl", repo_type="model"), encoding="utf-8") if line.strip()]
    gate_texts = [r["text"] for r in recs]
    y_gate = np.asarray([int(r["label"]) for r in recs], dtype=np.int64)
    genre_gate = np.asarray([r["genre"] for r in recs])
    group_gate = np.asarray([int(hashlib.sha1(str(r["pair_id"]).encode()).hexdigest()[:6], 16) for r in recs], dtype=np.int64)

    det_dir = DET_ID if os.path.isdir(DET_ID) else snapshot_download(
        DET_ID, allow_patterns=["config.json", "model.safetensors", "tokenizer*", "spm.model"]
    )
    model, tok = load_model_and_tokenizer(det_dir)
    model.to(device).eval()

    cache_path = os.path.join(ROOT, CACHE)
    rows_path = os.path.join(ROOT, ROWS_CACHE)
    if os.path.isfile(cache_path):
        with np.load(cache_path, allow_pickle=True) as data:
            texts_b, probs_b, emb_b, lab_b, grp_b, dom_b, slice_b = (
                list(data["texts"]), data["probs"], data["embeddings"], data["labels"], data["groups"], data["domain"], data["slice"]
            )
        print("[bank] loaded %d cached slice rows with embeddings" % len(texts_b), flush=True)
    else:
        if os.path.isfile(rows_path):
            # The data build streams ~20 min of sources; the chosen rows are worth keeping even
            # when the embedding pass has to be redone.
            blob = json.load(open(rows_path, encoding="utf-8"))
            texts_b = blob["texts"]
            slice_b = np.asarray(blob["slice"])
            lab_b = np.asarray(blob["labels"], dtype=np.int64)
            grp_b = np.asarray(blob["groups"], dtype=np.int64)
            dom_b = np.asarray(blob["domain"])
            print("[bank] reusing %d cached bank rows, data build skipped" % len(texts_b), flush=True)
        else:
            bundle = build_datasets(cfg)
            chosen = pick_bank(bundle)
            texts_b = [s.text for _, s in chosen]
            slice_b = np.asarray([n for n, _ in chosen])
            lab_b = np.asarray([int(s.label) for _, s in chosen], dtype=np.int64)
            grp_b = np.asarray([group_id(s) for _, s in chosen], dtype=np.int64)
            dom_b = np.asarray([s.domain for _, s in chosen])
            with open(rows_path, "w", encoding="utf-8") as fh:
                json.dump({"texts": texts_b, "slice": list(slice_b), "labels": lab_b.tolist(),
                           "groups": grp_b.tolist(), "domain": list(dom_b)}, fh)
        probs_b, emb_b = embed(model, tok, device, texts_b, "bank-slices")
        np.savez_compressed(cache_path, texts=np.asarray(texts_b), probs=probs_b, embeddings=emb_b,
                            labels=lab_b, groups=grp_b, domain=dom_b, slice=slice_b)
        print("[bank] cached %d rows to %s" % (len(texts_b), cache_path), flush=True)

    gate_cache = os.path.join(ROOT, GATE_CACHE)
    if os.path.isfile(gate_cache):
        with np.load(gate_cache) as data:
            probs_g, emb_g = data["probs"], data["embeddings"]
        probs_g, emb_g = probs_g.astype(np.float64), emb_g.astype(np.float64)
        print("[bank] reusing cached gate embeddings", flush=True)
    else:
        probs_g, emb_g = embed(model, tok, device, gate_texts, "gate")
        np.savez_compressed(gate_cache, probs=probs_g, embeddings=emb_g)

    X_bank = build_static(probs_b, texts_b)
    strat_bank = stratify(lab_b, dom_b)
    news, revs = genre_gate == "news", genre_gate == "reviews"

    def score_queries(cols, fusion, texts_q, probs_q, emb_q, groups_q, labels_q):
        """Query rows scored against the bank only; they are never their own neighbours."""
        V = vector_features(np.vstack([emb_b, emb_q]), np.concatenate([lab_b, labels_q]),
                            np.concatenate([grp_b, groups_q]),
                            query_idx=np.arange(len(texts_q)) + len(lab_b), ref_idx=np.arange(len(lab_b)),
                            k=K_NEIGHBOURS)
        return fusion.predict(assemble(cols, build_static(probs_q, texts_q), V))

    print("\n=== gate scored by a fusion that never saw a gate row ===")
    print(fmt("detector", probs_g, y_gate))
    print("  " + fmt("detector/news", probs_g, y_gate, news))
    print("  " + fmt("detector/reviews", probs_g, y_gate, revs))

    probe = probe_rows(PROBE_TXT)
    probe_probs = probe_emb = None
    if probe:
        probe_probs, probe_emb = embed(model, tok, device, [t for _, t in probe], "probe")
        print("[bank] probe texts embedded: %d" % len(probe), flush=True)
    probe_group = np.arange(len(probe)) + int(grp_b.max()) + 1 if probe else np.zeros(0, dtype=np.int64)

    saved = {"y": y_gate, "genre": genre_gate, "detector": probs_g}
    for name, cols in SETS.items():
        oof_bank, _, fusion, _ = oof_fusion(cols, X_bank, emb_b, lab_b, grp_b, strat_bank, k=K_NEIGHBOURS, seed=SEED)
        # deployment threshold: the 1%-FPR point on the bank's own out-of-fold scores, never on
        # the gate. Calibrating on the gate leaks the test population's shape into the cut.
        _, thr_bank = operating(oof_bank, lab_b, 0.01)
        fused = score_queries(cols, fusion, gate_texts, probs_g, emb_g, group_gate, y_gate)
        saved[name] = fused
        saved[name + "_threshold"] = np.asarray([thr_bank])
        print(fmt(name, fused, y_gate))
        print("  " + fmt(name + "/news", fused, y_gate, news))
        print("  " + fmt(name + "/reviews", fused, y_gate, revs))
        _, b_thr_gate = operating(probs_g, y_gate, 0.01)
        print("  bank-calibrated thr %.4f (deployment): catches %d/%d AI, %d human false alarms | "
              "gate-recalibrated catches %d/%d (optimistic)"
              % (thr_bank, int((fused >= thr_bank)[y_gate == 1].sum()), int((y_gate == 1).sum()),
                 int((fused >= thr_bank)[y_gate == 0].sum()),
                 int((fused >= operating(fused, y_gate, 0.01)[1])[y_gate == 1].sum()), int((y_gate == 1).sum())))
        print("  detector at its own gate 1%% point %.4f catches %d/%d"
              % (b_thr_gate, int((probs_g >= b_thr_gate)[y_gate == 1].sum()), int((y_gate == 1).sum())))

        if probe_probs is not None:
            p_fused = score_queries(cols, fusion, [t for _, t in probe], probe_probs, probe_emb,
                                    probe_group, np.ones(len(probe), dtype=np.int64))
            print("  explainers flagged: detector %d/%d (own gate 1%% thr), fusion %d/%d at bank-calibrated "
                  "%.4f, %d/%d at 0.5"
                  % (int((probe_probs >= b_thr_gate).sum()), len(probe),
                     int((p_fused >= thr_bank).sum()), len(probe), thr_bank,
                     int((p_fused >= 0.5).sum()), len(probe)))
            for i, (key, text) in enumerate(probe):
                print("    %-14s %6d det %.4f fusion %.4f" % (key, len(text), probe_probs[i], p_fused[i]))

    np.savez_compressed(
        os.path.join(ROOT, "hybrid_gate_externalbank.npz"),
        **{re.sub(r"\W", "_", k): v for k, v in saved.items()}
    )
    print("\nsaved hybrid_gate_externalbank.npz (gate scores from a bank it never contributed to)")


if __name__ == "__main__":
    main()
