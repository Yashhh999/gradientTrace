import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from gradienttrace.hybrid import (
    DETECTOR_NAMES,
    FEATURES,
    STYLO_NAMES,
    VECTOR_NAMES,
    Fusion,
    assemble,
    build_static,
    columns_for,
    oof_fusion,
    stratify,
    stylometry,
    vector_features,
)

HUMAN = "The train left early, so we walked home along the river and talked about nothing."
AI = "It is crucial to delve into the landscape of modern rail travel; furthermore, it offers seamless benefits."


def test_stylometry_complete_and_stable():
    a, b = stylometry(HUMAN), stylometry(HUMAN)
    assert sorted(a) == sorted(STYLO_NAMES), "feature names drifted"
    assert all(np.isfinite(list(a.values()))), "non-finite feature"
    assert a == b, "not deterministic"
    assert all(v == 0.0 for v in stylometry("").values())
    assert stylometry(AI)["llm_marker_rate"] > stylometry(HUMAN)["llm_marker_rate"]


def test_vector_features_block_pairs():
    embeddings = np.array([[1.0, 0.0], [0.999, 0.01], [0.6, 0.8], [0.8, 0.6]])
    labels = np.array([0, 1, 1, 0])
    # rows 0 and 1 are one prompt's human/AI pair: same topic, so they must never be neighbours
    paired_groups = np.array([0, 0, 1, 2])
    out = vector_features(embeddings, labels, paired_groups, np.arange(4), np.arange(4), k=1)
    assert out.shape == (4, len(VECTOR_NAMES)) and np.all(np.isfinite(out))
    margin = VECTOR_NAMES.index("knn_margin")
    assert out[0, VECTOR_NAMES.index("knn_ai")] < 0.7, "pair partner leaked as a neighbour"
    assert out[0, margin] > 0, "a human-side row should sit nearer the human cluster"
    free = vector_features(embeddings, labels, np.arange(4), np.arange(4), np.arange(4), k=1)
    assert free[0, VECTOR_NAMES.index("knn_ai")] > 0.99, "without groups the near-twin must be found"
    assert free[0, margin] < out[0, margin], "unblocked, the AI twin drives the margin the wrong way"


def test_vector_features_without_one_class():
    out = vector_features(np.eye(3), np.array([0, 0, 0]), np.arange(3), np.arange(3), np.arange(3))
    assert np.all(np.isfinite(out))
    assert np.all(out[:, VECTOR_NAMES.index("knn_human_frac")] == 0.5)


def test_oof_fusion_is_fold_safe_and_finite():
    rng = np.random.default_rng(0)
    n = 120
    human = rng.normal(-1.0, 0.4, size=(n // 2, 8))
    ai = rng.normal(1.0, 0.4, size=(n // 2, 8))
    embeddings = np.vstack([human, ai])
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    groups = np.arange(n)
    X_static = build_static(0.5 + 0.2 * (y - 0.5) + rng.normal(0, 0.02, n), ["a b c d e"] * n)
    for names in (DETECTOR_NAMES, STYLO_NAMES, VECTOR_NAMES, FEATURES):
        oof, coefs, fusion, X_all = oof_fusion(names, X_static, embeddings, y, groups, stratify(y, ["news"] * n))
        assert oof.shape == (n,) and np.all((oof >= 0) & (oof <= 1))
        assert coefs.shape[1] == len(names) and X_all.shape == (n, len(names))
        assert np.all(np.isfinite(fusion.predict(X_all)))
        assert fusion.predict(X_all).shape == (n,)


def test_assemble_rejects_unknown_names():
    try:
        assemble(["not_a_feature"], np.zeros((3, 1)), np.zeros((3, 0)))
    except KeyError:
        return
    raise AssertionError("unknown feature name should raise")


def test_fusion_json_roundtrip():
    X = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 0.5], [1.5, 2.5]])
    fusion = Fusion(columns_for(["det_logit", "knn_margin"]), [1.0, 1.0], [2.0, 2.0], [0.5, -0.25], 0.1, 0.42)
    scores = fusion.predict(X)
    restored = Fusion.from_json(fusion.to_json())
    assert np.allclose(scores, restored.predict(X))
    assert restored.threshold == 0.42
    assert FEATURES[0] == "det_logit" and len(FEATURES) == 1 + len(STYLO_NAMES) + len(VECTOR_NAMES)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("hybrid tests OK")
