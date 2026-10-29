import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from gradienttrace.metrics import binary_detection_metrics, sliced_report


def test_perfect_separation():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.6, 0.9])
    m = binary_detection_metrics(y, p)
    assert abs(m["auroc"] - 1.0) < 1e-9
    assert abs(m["auprc"] - 1.0) < 1e-9
    assert m["accuracy"] == 1.0
    assert m["tpr_at_fpr_0.01"] == 1.0
    assert 0.0 <= m["ece"] <= 1.0


def test_single_class_is_nan():
    m = binary_detection_metrics(np.array([1, 1, 1]), np.array([0.2, 0.5, 0.9]))
    assert m["n"] == 3
    assert np.isnan(m["auroc"])


def test_sliced_report_keys():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.2, 0.8, 0.4, 0.7])
    report = sliced_report(y, p, {"generator": np.array(["human", "a", "human", "a"])})
    assert "overall" in report
    assert "generator:a" in report
    assert report["generator:a"]["n"] == 2


if __name__ == "__main__":
    test_perfect_separation()
    test_single_class_is_nan()
    test_sliced_report_keys()
    print("metrics tests OK")
