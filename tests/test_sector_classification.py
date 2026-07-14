"""Characterization / smoke tests for utils/sector_classification.py."""

import re

import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression

from stock_embeddings.evaluation import get_sector_score


def _blobs(seed=0):
    # 3 well-separated classes, 90 samples (30/class) -> plenty for SMOTE
    # (needs >5/class) and StratifiedKFold(n_splits=5).
    X, y = make_blobs(
        n_samples=90, centers=3, n_features=5, cluster_std=0.5, random_state=seed
    )
    sectors = [f"SECTOR_{label}" for label in y]
    return X, sectors


def _parse_accuracy(captured_out):
    m = re.search(r"Accuracy Score:\s*([0-9.]+)", captured_out)
    assert m is not None, f"No accuracy printed in:\n{captured_out}"
    return float(m.group(1))


def test_get_sector_score_prints_all_scores_smote_true(capsys):
    X, sectors = _blobs()
    # LogisticRegression keeps it fast; separable blobs -> perfect scores.
    get_sector_score(
        X, sectors, classifier=LogisticRegression(max_iter=500), smote=True
    )
    out = capsys.readouterr().out
    for label in ("Precision Score:", "Recall Score:", "F1 Score:", "Accuracy Score:"):
        assert label in out
    assert _parse_accuracy(out) >= 0.9


def test_get_sector_score_smote_false(capsys):
    X, sectors = _blobs()
    get_sector_score(
        X, sectors, classifier=LogisticRegression(max_iter=500), smote=False
    )
    out = capsys.readouterr().out
    assert "Accuracy Score:" in out
    assert _parse_accuracy(out) >= 0.9


def test_get_sector_score_top_k(capsys):
    X, sectors = _blobs()
    clf = LogisticRegression(max_iter=500)  # supports predict_proba
    get_sector_score(X, sectors, classifier=clf, smote=False, top_k_accuracy=True, k=2)
    out = capsys.readouterr().out
    assert "Accuracy Score Top-2:" in out


@pytest.mark.filterwarnings("ignore")
def test_get_sector_score_default_svc_path(capsys):
    # Exercise the default classifier (SVC rbf, probability=True) once. Small data
    # keeps it quick.
    X, sectors = _blobs()
    get_sector_score(X, sectors)  # default classifier + smote=True
    out = capsys.readouterr().out
    assert "Accuracy Score:" in out
    assert _parse_accuracy(out) >= 0.9
