"""Characterization tests for utils/window_context.py."""

import numpy as np
import pytest
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr

from utils.window_context import (
    Euclidean,
    ICCBRMetric,
    Pearson,
    extract_indices,
    get_target_context_sets,
)


# --------------------------------------------------------------------------- #
# Pearson
# --------------------------------------------------------------------------- #
def test_pearson_compute_matches_scipy():
    rng = np.random.default_rng(0)
    v1 = rng.normal(size=8)
    v2 = rng.normal(size=8)
    metric = Pearson()
    assert metric.similarity is True
    assert np.isclose(metric.compute(v1, v2), pearsonr(v1, v2)[0])


def test_pearson_pairwise_matches_numpy():
    rng = np.random.default_rng(1)
    matrix = rng.normal(size=(5, 8))  # rows = time series
    metric = Pearson()
    assert np.allclose(metric.compute_matrix_pairwise(matrix), np.corrcoef(matrix))


# --------------------------------------------------------------------------- #
# Euclidean
# --------------------------------------------------------------------------- #
def test_euclidean_compute_vectorized():
    metric = Euclidean()  # vectorized=True by default
    assert metric.similarity is False
    query = np.array([0.0, 0.0, 0.0])
    others = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 5.0]])
    # distances from origin: 5 and 5
    result = metric.compute(query, others)
    assert result.shape == (2,)
    assert np.allclose(result, [5.0, 5.0])


def test_euclidean_compute_non_vectorized():
    metric = Euclidean(vectorized=False)
    a = np.array([1.0, 2.0])
    b = np.array([4.0, 6.0])
    assert np.isclose(metric.compute(a, b), 5.0)  # 3-4-5 triangle


def test_euclidean_pairwise_matches_scipy():
    rng = np.random.default_rng(2)
    matrix = rng.normal(size=(6, 4))
    metric = Euclidean()
    ref = squareform(pdist(matrix, "euclidean"))
    assert np.allclose(metric.compute_matrix_pairwise(matrix), ref)


# --------------------------------------------------------------------------- #
# ICCBRMetric
# --------------------------------------------------------------------------- #
def _iccbr_reference(v1, v2, w=0.5):
    cumprod_euclidean = abs(np.cumprod(1 + v1)[-1] - np.cumprod(1 + v2)[-1])
    nmc = (v1 @ v2) / (np.sqrt(np.sum(v1**2)) * np.sqrt(np.sum(v2**2)))
    return w / (1 + cumprod_euclidean) + (1 - w) * nmc


def test_iccbr_compute_matches_reference():
    v1 = np.array([0.01, 0.02, -0.01, 0.03])
    v2 = np.array([0.02, 0.01, 0.0, 0.02])
    metric = ICCBRMetric()  # w=0.5
    assert metric.similarity is True
    assert np.isclose(metric.compute(v1, v2), _iccbr_reference(v1, v2, 0.5))


def test_iccbr_compute_respects_weight():
    v1 = np.array([0.01, 0.02, -0.01])
    v2 = np.array([0.02, 0.01, 0.0])
    metric = ICCBRMetric(w=0.3)
    assert np.isclose(metric.compute(v1, v2), _iccbr_reference(v1, v2, 0.3))


def test_iccbr_pairwise_matches_elementwise_and_diagonal():
    rng = np.random.default_rng(3)
    matrix = rng.normal(0, 0.02, size=(4, 6))  # rows = time series
    metric = ICCBRMetric()
    pw = metric.compute_matrix_pairwise(matrix)
    n = matrix.shape[0]
    assert pw.shape == (n, n)
    # diagonal is hard-coded to 1.0 and matrix is symmetric
    assert np.allclose(np.diag(pw), 1.0)
    assert np.allclose(pw, pw.T)
    ref = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            ref[i, j] = ref[j, i] = metric.compute(matrix[i], matrix[j])
    assert np.allclose(pw, ref)


# --------------------------------------------------------------------------- #
# extract_indices
# --------------------------------------------------------------------------- #
# A hand-built matrix. Rows 0/1 are mutually most-similar / closest; rows 2/3 too.
_M = np.array(
    [
        [1.0, 0.9, 0.1, 0.2],
        [0.9, 1.0, 0.3, 0.1],
        [0.1, 0.3, 1.0, 0.8],
        [0.2, 0.1, 0.8, 1.0],
    ]
)


def test_extract_indices_similarity_picks_largest():
    # similarity metric: diagonal is masked to -1, argpartition picks the largest.
    result = extract_indices(_M, context_size=1, metric_class=Pearson())
    got = {i: ctx[0] for i, ctx in result}
    assert got == {0: 1, 1: 0, 2: 3, 3: 2}


def test_extract_indices_distance_picks_smallest():
    # distance metric: multiplier=-1 flips ordering so the *smallest* off-diagonal
    # entry is selected (nearest neighbor). Off-diagonal values here are all < 1 so
    # the self entry (masked to -1) is never chosen.
    result = extract_indices(_M, context_size=1, metric_class=Euclidean())
    got = {i: ctx[0] for i, ctx in result}
    # row0 smallest off-diag is 0.1 at col2; row1: 0.1 at col3; etc.
    assert got == {0: 2, 1: 3, 2: 0, 3: 1}


def test_extract_indices_excludes_self_and_correct_size():
    result = extract_indices(_M, context_size=2, metric_class=Pearson())
    assert len(result) == 4
    for i, ctx in result:
        assert len(ctx) == 2
        assert i not in ctx  # self is masked out via the diagonal


# --------------------------------------------------------------------------- #
# get_target_context_sets (end-to-end over windows)
# --------------------------------------------------------------------------- #
def test_get_target_context_sets_known_neighbors_euclidean():
    # 4 time series: series 0 & 1 nearly identical, series 2 & 3 nearly identical.
    # Over any window each series' nearest neighbor is its pair partner.
    rng = np.random.default_rng(4)
    base_a = rng.normal(0, 0.01, size=20)
    base_b = rng.normal(0, 0.01, size=20)
    X = np.vstack(
        [
            base_a,
            base_a + 1e-4,
            base_b + 5.0,
            base_b + 5.0 + 1e-4,
        ]
    )
    sets = get_target_context_sets(
        X, Euclidean(), window_length=5, stride=5, context_size=1, verbose=False
    )
    # Every set must map a series to its (unique) near-duplicate partner.
    partner = {0: 1, 1: 0, 2: 3, 3: 2}
    assert len(sets) > 0
    for tgt, ctx in sets:
        assert ctx[0] == partner[tgt]


def test_get_target_context_sets_structure():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(6, 30))
    cs = 2
    sets = get_target_context_sets(
        X, Pearson(), window_length=6, stride=6, context_size=cs, verbose=False
    )
    for tgt, ctx in sets:
        assert 0 <= tgt < 6
        assert len(ctx) == cs
        assert tgt not in ctx
