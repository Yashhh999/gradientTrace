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
    DETECTOR_NAMES,
    FEATURES,
    STYLO_NAMES,
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
    bos_token_id = 101
    eos_token_id = 102

    def _ids(self, text, truncation, max_length):
        ids = [(sum(ord(ch) for ch in w) % (self.vocab - 3)) + 3 for w in text.split()]
        return ids[: max(0, max_length - 2)] if truncation else ids

    def __call__(self, texts, truncation=True, max_length=64, padding=True, return_tensors="pt", add_special_tokens=True):
        if isinstance(texts, str):
            texts = [texts]
        rows = []
        for t in texts:
            ids = self._ids(t, truncation, max_length)
            rows.append([self.bos_token_id, *ids, self.eos_token_id] if add_special_tokens else ids)
        if return_tensors is None:
            return {"input_ids": rows, "attention_mask": [[1] * len(r) for r in rows]}
        width = max(len(r) for r in rows)
        return {
            "input_ids": torch.tensor([r + [0] * (width - len(r)) for r in rows]),
            "attention_mask": torch.tensor([[1] * len(r) + [0] * (width - len(r)) for r in rows]),
        }

    def pad(self, example, padding=True, return_tensors="pt"):
        rows = list(example["input_ids"])
        self.seen = getattr(self, "seen", 0) + sum(len(r) for r in rows)
        width = max(len(r) for r in rows)
        return {
            "input_ids": torch.tensor([r + [0] * (width - len(r)) for r in rows]),
            "attention_mask": torch.tensor([[1] * len(r) + [0] * (width - len(r)) for r in rows]),
        }


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


def test_chunked_scoring_uses_the_tail():
    if torch is None:
        print("skip: torch not installed")
        return
    from gradienttrace.data import Sample
    from gradienttrace.evaluate import predict_probs, predict_probs_and_embeddings

    model, tok = tiny_detector()
    device = torch.device("cpu")
    head = "The quick brown fox jumps over the lazy dog"
    tail = ("and the glacier melted slowly into the sea where the ships waited patiently "
            "for the tide to change its mind")
    samples = [Sample(text=head, label=0, generator="x", source="test"),
               Sample(text=head + " " + tail, label=1, generator="x", source="test")]
    tok.seen = 0
    probs, vectors = predict_probs_and_embeddings(model, tok, samples, device, max_length=24, batch_size=2)
    chunked_tokens = tok.seen
    tok.seen = 0
    unchunked, _ = predict_probs_and_embeddings(
        model, tok, samples, device, max_length=24, batch_size=2, chunk=False
    )
    window_tokens = tok.seen
    assert probs.shape == (2,) and vectors.shape == (2, 16)
    assert np.isfinite(probs).all()
    # a short text goes through one window, identical to the truncated path
    assert abs(probs[0] - predict_probs(model, tok, [samples[0]], device, max_length=24, batch_size=1)[0]) < 1e-5
    # chunk=False must reproduce truncation exactly
    unchunked, _ = predict_probs_and_embeddings(
        model, tok, samples, device, max_length=24, batch_size=2, chunk=False
    )
    assert abs(unchunked[1] - predict_probs(model, tok, [samples[1]], device, max_length=24, batch_size=1)[0]) < 1e-5
    # and chunking must actually forward tokens from the tail, which the window count proves
    # without depending on how discriminative a randomly-initialised model happens to be
    lens = [len(tok(s.text, truncation=False)["input_ids"][0]) for s in samples]
    full_tokens = sum(l - 2 for l in lens)
    window = 24 - int(tok.bos_token_id is not None) - int(tok.eos_token_id is not None)
    content = [l - 2 for l in lens]
    windows = sum(max(1, -(-c // window)) for c in content)
    assert full_tokens > window, "test text must exceed one window for this assertion to mean anything"
    assert chunked_tokens > window_tokens, "chunking must forward the tail; truncation must stop at one window"
    assert chunked_tokens == full_tokens + 2 * windows, (chunked_tokens, full_tokens, windows)


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
        full, probe_probs, made = scorer.features(gate[:5])
        assert made == FEATURES, "a banked scorer must report the columns it built"
        assert full.shape == (5, len(FEATURES)), full.shape
        X = full[:, columns_for(names)]
        with np.load(f"{tmp}/bank.npz") as loaded:
            assert loaded["embeddings"].shape == (24, 16) and loaded["labels"].sum() == 12
        again = scorer.score(gate[:5])
        assert np.allclose(again["fused"], fusion.predict(X), atol=1e-9), "scorer lost the loaded weights"
        assert again["threshold"] == 0.6
        assert np.isfinite(again["fused"]).all() and np.isfinite(probe_probs).all()
        assert Fusion.from_json(fusion.to_json()).names == names

        # the shipped fusion has no vector columns, so it must work without touching the bank
        bankless_names = ["det_logit", "log_words", "ttr"]
        _, _, bankless, _ = oof_fusion(bankless_names, X_static, embeddings.astype(np.float64),
                                       labels, groups, stratify(labels, genres))
        bankless.threshold = 0.5
        save_fusion(tmp, bankless, embeddings, labels, groups, "tiny-detector", k=4)
        plain = HybridScorer("ignored", tmp, device=device, batch_size=6, detector=(model, tok))
        full2, _, made2 = plain.features(gate[:5])
        assert made2 == DETECTOR_NAMES + STYLO_NAMES, "vector block should be skipped without a bank"
        assert full2.shape == (5, len(DETECTOR_NAMES) + len(STYLO_NAMES))
        assert np.isfinite(plain.score(gate[:5])["fused"]).all()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("hybrid inference tests OK")
