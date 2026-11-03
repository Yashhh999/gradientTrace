"""Inference-path tests for the hybrid: embedding extraction, fold-safe fusion and the saved
artifact round trip. Uses a tiny randomly-initialised encoder so nothing is downloaded; the
numbers are meaningless by design, the code paths are not.

Skipped when torch is absent (CI installs only the lightweight deps).
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

try:
    import torch
    from transformers import BertConfig
except ImportError:
    torch = None

from gradienttrace.hybrid import (  # noqa: E402
    FEATURES,
    Fusion,
    HybridScorer,
    build_static,
    columns_for,
    oof_fusion,
    save_fusion,
    stratify,
)


class TinyTokenizer:
    """Stand-in for the fast tokenizer: whitespace-split ids, dynamic padding."""

    vocab = 200

    def __call__(self, texts, truncation=True, max_length=64, padding=True, return_tensors="pt"):
        if isinstance(texts, str):
            texts = [texts]
        rows = [
            [(sum(ord(ch) for ch in w) % (self.vocab - 3)) + 3 for w in t.split()[: max_length - 2]]
            for t in texts
        ]
        rows = [[1] + r + [2] for r in rows]
        width = max(len(r) for r in rows)
        ids, mask = [], []
        for r in rows:
            ids.append(r + [0] * (width - len(r)))
            mask.append([1] * len(r) + [0] * (width - len(r)))
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(mask)}


def tiny_detector(pooling="cls"):
    from gradienttrace.model import AIDetector

    config = BertConfig(
        vocab_size=TinyTokenizer.vocab,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=64,
    )
    config.pooling = pooling
    torch.manual_seed(0)
    model = AIDetector(config)
    return model.eval(), TinyTokenizer()


def texts(n=24):
    return ["The %s river ran quickly, so we walked home and talked." % (i % 5) for i in range(n)]


def test_embeddings_match_forward_path():
    if torch is None:
        print("skip: torch not installed")
        return
    from gradienttrace.data import Sample
    from gradienttrace.evaluate import predict_probs, predict_probs_and_embeddings

    model, tok = tiny_detector()
    samples = [Sample(text=t, label=1, generator="x", source="test") for t in texts()]
    device = torch.device("cpu")
    plain = predict_probs(model, tok, samples, device, batch_size=5)
    probs, vectors = predict_probs_and_embeddings(model, tok, samples, device, batch_size=5)
    assert np.allclose(plain, probs, atol=1e-6), "encode()/pool() refactor drifted from forward()"
    assert vectors.shape == (len(samples), 16), vectors.shape
    assert np.isfinite(vectors).all()
    assert np.linalg.norm(vectors, axis=1).min() > 0, "pooled vectors are all zero"


def test_mean_pooling_path():
    if torch is None:
        return
    from gradienttrace.data import Sample
    from gradienttrace.evaluate import predict_probs_and_embeddings

    model, tok = tiny_detector(pooling="mean")
    device = torch.device("cpu")
    probs, vectors = predict_probs_and_embeddings(
        model, tok, [Sample(text=t, label=1, generator="x", source="test") for t in texts(6)], device, batch_size=3
    )
    assert probs.shape == (6,) and vectors.shape == (6, 16)


def test_save_and_score_roundtrip():
    if torch is None:
        return
    model, tok = tiny_detector()
    device = torch.device("cpu")
    gate = texts(24)
    labels = np.array([0] * 12 + [1] * 12)
    groups = np.arange(24)
    genres = np.asarray(["news"] * 12 + ["reviews"] * 12)

    from gradienttrace.data import Sample
    from gradienttrace.evaluate import predict_probs_and_embeddings

    probs, embeddings = predict_probs_and_embeddings(
        model, tok, [Sample(text=t, label=1, generator="x", source="gate") for t in gate], device, batch_size=6
    )
    X_static = build_static(probs, gate)
    names = ["det_logit", "log_words", "ttr", "knn_margin", "centroid_margin"]
    oof, _, fusion, _ = oof_fusion(names, X_static, embeddings.astype(np.float64), labels, groups, stratify(labels, genres))
    fusion.threshold = 0.6

    with tempfile.TemporaryDirectory() as tmp:
        save_fusion(tmp, fusion, embeddings, labels, groups, "tiny-detector", k=4)
        scorer = HybridScorer("ignored", tmp, device=device, batch_size=6, detector=(model, tok))
        full, probe_probs = scorer.features(gate[:5])
        assert full.shape[0] == 5 and full.shape[1] == len(FEATURES), full.shape
        X = full[:, columns_for(names)]
        with np.load(f"{tmp}/bank.npz") as loaded:
            assert loaded["embeddings"].shape == (24, 16) and loaded["labels"].sum() == 12
        again = scorer.score(gate[:5])
        assert np.allclose(again["fused"], fusion.predict(X), atol=1e-9), "scorer lost the loaded weights"
        assert again["threshold"] == 0.6
        assert np.isfinite(again["fused"]).all() and np.isfinite(probe_probs).all()
        assert Fusion.from_json(fusion.to_json()).names == names


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("hybrid inference tests OK")
