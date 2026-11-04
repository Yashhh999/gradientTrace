"""Stylometric and embedding-vector features for the hybrid scorer.

Two signals that the detector's scalar logit does not carry:

* ``stylometry`` — hand-counted surface statistics (function words, punctuation classes,
  sentence-length burstiness, lexical diversity, formatting tells). Cheap, CPU-only and
  interpretable; also the block most vulnerable to a paraphrase attack.
* ``vector_features`` — geometry of the detector's *own* pooled representation: how close a text
  sits to labelled human versus labelled AI examples (k-NN margin and centroid distance). The
  logit collapses that space to one number, so neighbours can disagree with the head.

Both fuse with the logit by logistic regression. One rule matters for correctness: vector
features for a scored row may only use references from the training side of the split, and never
a reference sharing its group (the same prompt's human/AI pair), because that is topic similarity
rather than style.
"""

from __future__ import annotations

import json
import math
import re

import numpy as np

FUNCTION_WORDS = frozenset(
    """a an the and but or nor so yet of in on at by for with without to from into onto upon
    about above below through during before after between among as than that this these those
    there here it its he she they them his her their you your we our i me my mine us is are was
    were be been being have has had having do does did will would shall should can could may
    might must not no all any each every some such only own same other another much many more
    most less least few""".split()
)
PRONOUNS = frozenset(
    "i me my mine we us our ours you your yours he him his she her hers it its they them their theirs".split()
)
DETERMINERS = frozenset("a an the this that these those every each some no any".split())
PREPOSITIONS = frozenset(
    "of in on at by for with without to from into onto upon about above below through during before after between among as per via".split()
)
CONJUNCTIONS = frozenset("and but or nor so yet although because while whereas since unless whether though".split())
AUXILIARIES = frozenset(
    "is are was were be been being have has had having do does did will would shall should can could may might must".split()
)
CONNECTIVES = frozenset(
    "however moreover furthermore therefore thus consequently additionally meanwhile nevertheless".split()
)

STYLO_NAMES = [
    "mean_word_len", "long_word_rate", "syll_per_word", "flesch",
    "function_rate", "pronoun_rate", "determiner_rate", "preposition_rate", "conjunction_rate",
    "auxiliary_rate", "connective_rate", "digit_rate", "uppercase_ratio",
    "allcaps_rate", "contraction_rate", "comma_rate", "period_rate", "colon_rate", "quote_rate",
    "parens_rate", "dash_rate", "ellipsis_rate", "exclaim_rate", "question_rate",
    "markdown_rate", "newline_rate", "sent_std_words", "ttr", "hapax_rate",
    "trigram_repeat",
]
VECTOR_NAMES = ["knn_human", "knn_ai", "knn_margin", "knn_human_frac", "centroid_margin"]
DETECTOR_NAMES = ["det_logit"]
FEATURES = DETECTOR_NAMES + STYLO_NAMES + VECTOR_NAMES
K_NEIGHBOURS = 8
_BLOCKED = -1e30


def _words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text)


def _syllables(word: str) -> int:
    word = word.lower()
    n = len(re.findall(r"[aeiouy]+", word))
    if n > 1 and word.endswith("e") and not word.endswith(("le", "ee")):
        n -= 1
    return max(1, n)


def _sentences(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"[.!?]+[\s\"')\]]|(?<=[.!?])\n", text) if p.strip()]


def stylometry(text: str) -> dict[str, float]:
    """Surface statistics; every *_rate is per 100 words. Deterministic and network-free."""
    words = _words(text)
    n = len(words)
    if n == 0:
        return {name: 0.0 for name in STYLO_NAMES}
    lower = [w.lower() for w in words]
    sents = _sentences(text) or [""]
    sent_lengths = [len(_words(s)) for s in sents]
    syllables = sum(_syllables(w) for w in words)
    unique = set(lower)
    counts: dict[str, int] = {}
    for w in lower:
        counts[w] = counts.get(w, 0) + 1
    trigrams = [tuple(lower[i : i + 3]) for i in range(max(0, n - 2))]
    seen: set[tuple] = set()
    repeats = 0
    for triple in trigrams:
        if triple in seen:
            repeats += 1
        seen.add(triple)
    rate = lambda count: 100.0 * count / n  # noqa: E731
    in_lines = lambda items: sum(1 for w in lower if w in items)  # noqa: E731
    alpha = sum(1 for c in text if c.isalpha()) or 1

    return {
        "mean_word_len": sum(len(w) for w in words) / n,
        "long_word_rate": rate(sum(1 for w in words if len(w) > 6)),
        "syll_per_word": syllables / n,
        "flesch": 206.835 - 1.015 * (n / len(sents)) - 84.6 * (syllables / n),
        "function_rate": rate(in_lines(FUNCTION_WORDS)),
        "pronoun_rate": rate(in_lines(PRONOUNS)),
        "determiner_rate": rate(in_lines(DETERMINERS)),
        "preposition_rate": rate(in_lines(PREPOSITIONS)),
        "conjunction_rate": rate(in_lines(CONJUNCTIONS)),
        "auxiliary_rate": rate(in_lines(AUXILIARIES)),
        "connective_rate": rate(in_lines(CONNECTIVES)),
        "digit_rate": rate(sum(1 for c in text if c.isdigit())),
        "uppercase_ratio": sum(1 for c in text if c.isupper()) / alpha,
        "allcaps_rate": rate(sum(1 for w in words if len(w) > 1 and w.isupper())),
        "contraction_rate": rate(sum(1 for w in words if "'" in w)),
        "comma_rate": rate(text.count(",")),
        "period_rate": rate(text.count(".")),
        "colon_rate": rate(text.count(":") + text.count(";")),
        "quote_rate": rate(sum(text.count(c) for c in "\"'“”‘’")),
        "parens_rate": rate(min(text.count("("), text.count(")"))),
        "dash_rate": rate(sum(text.count(c) for c in "-–—")),
        "ellipsis_rate": rate(text.count("...") + text.count("…")),
        "exclaim_rate": rate(text.count("!")),
        "question_rate": rate(text.count("?")),
        "markdown_rate": rate(
            sum(1 for line in text.splitlines() if re.match(r"^\s*(?:[-*•]|\d+\.|#+|\*\*)", line))
        ),
        "newline_rate": rate(text.count("\n")),
        "sent_std_words": float(np.std(sent_lengths)),
        "ttr": len(unique) / n,
        "hapax_rate": rate(sum(1 for c in counts.values() if c == 1)),
        "trigram_repeat": repeats / len(trigrams) if trigrams else 0.0,
    }


def stylometry_matrix(texts) -> np.ndarray:
    return np.asarray([[stylometry(t)[name] for name in STYLO_NAMES] for t in texts], dtype=np.float64)


def _unit(vectors: np.ndarray) -> np.ndarray:
    return vectors / np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-8, None)


def _topk_mean(scores: np.ndarray, k: int) -> np.ndarray:
    ordered = np.sort(scores, axis=1)[:, -max(1, k):]
    valid = ordered > _BLOCKED / 2
    means = np.nanmean(np.where(valid, ordered, np.nan), axis=1)
    return np.where(valid.any(axis=1), means, -1.0)


def _human_fraction(masked: np.ndarray, ref_labels: np.ndarray, k: int) -> np.ndarray:
    neighbours = np.argsort(-masked, axis=1)[:, : max(1, k)]
    gathered = np.take_along_axis(masked, neighbours, axis=1)
    valid = gathered > _BLOCKED / 2
    human = ((ref_labels[neighbours] == 0) & valid).sum(axis=1)
    total = np.maximum(valid.sum(axis=1), 1)
    return np.where(valid.any(axis=1), human / total, 0.5)


def vector_features(embeddings, labels, groups, query_idx, ref_idx, k: int = K_NEIGHBOURS) -> np.ndarray:
    """k-NN and centroid geometry of each query row, measured only against ``ref_idx``.

    References sharing a group with the query are blocked (same prompt, same topic), and the
    feature signs are oriented so that larger means *more human-like* placement.
    """
    embeddings = np.asarray(embeddings, dtype=np.float64)
    labels = np.asarray(labels)
    groups = np.asarray(groups, dtype=np.int64)
    queries = np.asarray(query_idx, dtype=np.int64)
    refs = np.asarray(ref_idx, dtype=np.int64)
    out = np.full((len(queries), len(VECTOR_NAMES)), np.nan)
    if len(queries) == 0:
        return out
    if len(refs) == 0 or not (labels[refs] == 0).any() or not (labels[refs] == 1).any():
        out[:, :2] = -1.0
        out[:, 2] = 0.0
        out[:, 3] = 0.5
        out[:, 4] = 0.0
        return out

    unit = _unit(embeddings)
    sim = unit[queries] @ unit[refs].T
    same = groups[queries][:, None] == groups[refs][None, :]
    blocked = np.where(same, _BLOCKED, sim)
    ref_labels = labels[refs]
    human = np.where(same | (ref_labels[None, :] != 0), _BLOCKED, sim)
    ai = np.where(same | (ref_labels[None, :] != 1), _BLOCKED, sim)

    centre_h = unit[refs[ref_labels == 0]].mean(axis=0)
    centre_a = unit[refs[ref_labels == 1]].mean(axis=0)
    dist_h = np.linalg.norm(unit[queries] - centre_h, axis=1)
    dist_a = np.linalg.norm(unit[queries] - centre_a, axis=1)

    knn_human = _topk_mean(human, k)
    knn_ai = _topk_mean(ai, k)
    out[:, 0] = knn_human
    out[:, 1] = knn_ai
    out[:, 2] = knn_human - knn_ai
    out[:, 3] = _human_fraction(blocked, ref_labels, k)
    out[:, 4] = dist_a - dist_h
    return out


def columns_for(names: list[str]) -> list[int]:
    return [FEATURES.index(name) for name in names]


def stratify(labels, genres) -> np.ndarray:
    return np.char.add(np.char.add(np.asarray(labels).astype(str), "_"), np.asarray(genres))


STATIC_NAMES = DETECTOR_NAMES + STYLO_NAMES


def build_static(det_probs, texts) -> np.ndarray:
    """The fold-independent columns: detector logit plus stylometry."""
    p = np.clip(np.asarray(det_probs, dtype=np.float64).ravel(), 1e-6, 1 - 1e-6)
    return np.column_stack([np.log(p / (1.0 - p)), stylometry_matrix(texts)])


def assemble(names, X_static, vectors) -> np.ndarray:
    """Pick the requested columns, taking vector features from a fold-specific matrix."""
    cols = []
    for name in names:
        if name in STATIC_NAMES:
            cols.append(X_static[:, STATIC_NAMES.index(name)])
        elif name in VECTOR_NAMES:
            cols.append(vectors[:, VECTOR_NAMES.index(name)])
        else:
            raise KeyError(f"unknown hybrid feature {name!r}")
    return np.column_stack(cols)


def oof_fusion(names, X_static, embeddings, labels, groups, strat, k: int = K_NEIGHBOURS, seed: int = 0, folds: int = 5):
    """Out-of-fold logistic fusion where vector features are rebuilt inside every fold.

    Vector features for a scored row may only see references from the training side, otherwise
    the k-NN columns leak the held-out labels and the reported gain is fiction.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    labels = np.asarray(labels)
    uses_vectors = any(name in VECTOR_NAMES for name in names)
    probs = np.zeros(len(labels), dtype=np.float64)
    coefs = []
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(X_static, strat):
        if uses_vectors:
            V_tr = vector_features(embeddings, labels, groups, tr, tr, k)
            V_te = vector_features(embeddings, labels, groups, te, tr, k)
        else:
            V_tr = V_te = np.zeros((len(tr), 0))
        X_tr, X_te = assemble(names, X_static[tr], V_tr), assemble(names, X_static[te], V_te)
        scaler = StandardScaler().fit(X_tr)
        clf = LogisticRegression(C=1.0, max_iter=4000).fit(scaler.transform(X_tr), labels[tr])
        probs[te] = clf.predict_proba(scaler.transform(X_te))[:, 1]
        coefs.append(clf.coef_[0])

    V_all = vector_features(embeddings, labels, groups, np.arange(len(labels)), np.arange(len(labels)), k) if uses_vectors else np.zeros((len(labels), 0))
    X_all = assemble(names, X_static, V_all)
    final_scaler = StandardScaler().fit(X_all)
    final = LogisticRegression(C=1.0, max_iter=4000).fit(final_scaler.transform(X_all), labels)
    return probs, np.vstack(coefs), Fusion.from_final(final_scaler, final, names), X_all


class Fusion:
    """Standardizer plus logistic weights over a named feature subset; serialises to JSON."""

    def __init__(self, names, mean, scale, coef, intercept, threshold=None):
        self.names = list(names)
        self.mean = np.asarray(mean, dtype=np.float64)
        self.scale = np.asarray(scale, dtype=np.float64)
        self.coef = np.asarray(coef, dtype=np.float64)
        self.intercept = float(intercept)
        self.threshold = threshold

    @classmethod
    def from_final(cls, scaler, clf, names, threshold=None):
        return cls(names, scaler.mean_, scaler.scale_, clf.coef_[0], clf.intercept_[0], threshold)

    def predict(self, X) -> np.ndarray:
        z = (np.asarray(X, dtype=np.float64) - self.mean) / np.clip(self.scale, 1e-9, None)
        return 1.0 / (1.0 + np.exp(-(z @ self.coef + self.intercept)))

    def to_json(self) -> dict:
        return {
            "names": self.names,
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "coef": self.coef.tolist(),
            "intercept": self.intercept,
            "threshold": self.threshold,
        }

    @classmethod
    def from_json(cls, blob: dict) -> "Fusion":
        return cls(
            blob["names"],
            np.asarray(blob["mean"], dtype=np.float64),
            np.asarray(blob["scale"], dtype=np.float64),
            np.asarray(blob["coef"], dtype=np.float64),
            blob["intercept"],
            blob.get("threshold"),
        )


def save_fusion(directory: str, fusion: Fusion, embeddings, labels, groups, detector_id: str, k: int = K_NEIGHBOURS):
    import os

    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "fusion.json"), "w", encoding="utf-8") as fh:
        json.dump(
            {"fusion": fusion.to_json(), "detector": detector_id, "k": k, "features": FEATURES},
            fh,
            indent=1,
        )
    np.savez_compressed(
        os.path.join(directory, "bank.npz"),
        embeddings=np.asarray(embeddings, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        groups=np.asarray(groups, dtype=np.int64),
    )


class HybridScorer:
    """Scores raw text: detector logit + stylometry + k-NN geometry against a labelled bank."""

    def __init__(self, detector_dir, fusion_dir, device=None, batch_size=16, detector=None):
        import os

        import torch

        from gradienttrace.evaluate import load_model_and_tokenizer

        blob = json.load(open(os.path.join(fusion_dir, "fusion.json"), encoding="utf-8"))
        self.fusion = Fusion.from_json(blob["fusion"])
        self.detector_id = blob["detector"]
        self.k = int(blob.get("k", K_NEIGHBOURS))
        with np.load(os.path.join(fusion_dir, "bank.npz")) as bank:
            self.bank_embeddings = bank["embeddings"].astype(np.float64)
            self.bank_labels = bank["labels"].copy()
            self.bank_groups = bank["groups"].copy()
        if not any(name in VECTOR_NAMES for name in self.fusion.names):
            self.bank_embeddings = self.bank_labels = self.bank_groups = None
        if detector is not None:
            self.model, self.tokenizer = detector
        else:
            self.model, self.tokenizer = load_model_and_tokenizer(detector_dir)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device).eval()
        self.batch_size = batch_size

    def features(self, texts):
        from gradienttrace.data import Sample
        from gradienttrace.evaluate import predict_probs_and_embeddings

        samples = [Sample(text=t, label=1, generator="x", source="hybrid") for t in texts]
        probs, vectors = predict_probs_and_embeddings(
            self.model, self.tokenizer, samples, self.device, batch_size=self.batch_size, desc="hybrid"
        )
        p = np.clip(probs.astype(np.float64), 1e-6, 1 - 1e-6)
        blocks = [np.log(p / (1 - p))[:, None], stylometry_matrix(texts)]
        if self.bank_embeddings is not None:
            embeddings = np.vstack([self.bank_embeddings, vectors.astype(np.float64)])
            labels = np.concatenate([self.bank_labels, np.ones(len(texts), dtype=np.int64)])
            # Each incoming text is its own group: it may borrow neighbours from the bank but never
            # from a row that shares its topic, and it is not its own neighbour.
            groups = np.concatenate([self.bank_groups, np.arange(len(texts)) + int(self.bank_groups.max()) + 1])
            blocks.append(
                vector_features(
                    embeddings, labels, groups,
                    query_idx=np.arange(len(self.bank_labels), len(self.bank_labels) + len(texts)),
                    ref_idx=np.arange(len(self.bank_labels)),
                    k=self.k,
                )
            )
        made = DETECTOR_NAMES + STYLO_NAMES + (VECTOR_NAMES if self.bank_embeddings is not None else [])
        return np.hstack(blocks), probs, made

    def score(self, texts) -> dict:
        full, probs, made = self.features(texts)
        fused = self.fusion.predict(full[:, [made.index(n) for n in self.fusion.names]])
        threshold = self.fusion.threshold if self.fusion.threshold is not None else 0.5
        return {"fused": fused, "detector": probs, "threshold": float(threshold), "flagged": fused >= threshold}
