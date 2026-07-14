"""Characterization tests for models/matrix_factorization.py (ported MF).

All synthetic, no network. Similarity matrices are checked against independent
numpy references; masking is checked for determinism; training is a seeded
smoke test (loss decreases, stays finite).
"""

import numpy as np
import pytest
import torch

from stock_embeddings.models.matrix_factorization import MatrixFactorization


def _X(n=10, length=25, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, length))


# --------------------------------------------------------------- construction
def test_forward_is_gram_matrix():
    torch.manual_seed(0)
    model = MatrixFactorization(6, 4)
    W = model.embeddings.weight.detach().numpy()
    out = model().detach().numpy()
    assert np.allclose(out, W @ W.T, atol=1e-6)
    assert out.shape == (6, 6)


def test_normalize_makes_unit_rows():
    torch.manual_seed(1)
    model = MatrixFactorization(8, 5, normalize=True)
    norms = model.embeddings.weight.detach().norm(dim=1).numpy()
    assert np.allclose(norms, np.ones(8), atol=1e-6)


# ------------------------------------------------------- correlation matrix
def test_correlation_matrix_matches_numpy_reference():
    X = _X()
    out = MatrixFactorization.get_correlation_matrix(X, scaled=False)
    ref = np.corrcoef(X)
    np.fill_diagonal(ref, np.nan)
    assert np.allclose(out.numpy(), ref, atol=1e-10, equal_nan=True)


def test_correlation_diagonal_is_nan():
    out = MatrixFactorization.get_correlation_matrix(_X(), scaled=False)
    assert torch.isnan(torch.diagonal(out)).all()


# --------------------------------------------------------- euclidean matrix
def test_euclidean_matrix_matches_reference():
    X = _X(seed=3)
    out = MatrixFactorization.get_euclidean_matrix(X, scaled=False, verbose=False)
    n = X.shape[0]
    ref = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            ref[i, j] = np.sqrt(((X[i] - X[j]) ** 2).sum())
    np.fill_diagonal(ref, np.nan)
    assert np.allclose(out.numpy(), ref, atol=1e-8, equal_nan=True)


def test_euclidean_return_similarity_is_negated():
    X = _X(seed=4)
    dist = MatrixFactorization.get_euclidean_matrix(X, scaled=False, verbose=False)
    sim = MatrixFactorization.get_euclidean_matrix(
        X, scaled=False, return_similarity=True, verbose=False
    )
    mask = ~torch.isnan(dist)
    assert torch.allclose(sim[mask], -dist[mask], atol=1e-8)


# ------------------------------------------------------------- scalings
def test_min_max_scale_range_within_pm1():
    X = _X(seed=5)
    out = MatrixFactorization.get_correlation_matrix(X, scaled="min_max")
    vals = out[~torch.isnan(out)]
    # off-diagonal min maps to -1; overall bounded by construction
    assert vals.min().item() >= -1 - 1e-6


def test_normal_scale_zero_mean_offdiagonal():
    X = _X(seed=6)
    out = MatrixFactorization.get_correlation_matrix(X, scaled="normal")
    off = out[~torch.isnan(out)]
    assert abs(off.mean().item()) < 1e-6


def test_truncate_clamps_to_pm2():
    X = _X(seed=7)
    out = MatrixFactorization.get_correlation_matrix(
        X, scaled="normal", truncate=True
    )
    off = out[~torch.isnan(out)]
    assert off.max().item() <= 2 + 1e-6
    assert off.min().item() >= -2 - 1e-6


# --------------------------------------------------------------- masking
def test_apply_random_mask_deterministic_with_seed():
    base = torch.tensor(np.corrcoef(_X(seed=8)))
    torch.manual_seed(99)
    a = MatrixFactorization.apply_random_mask(base.clone(), 0.4)
    torch.manual_seed(99)
    b = MatrixFactorization.apply_random_mask(base.clone(), 0.4)
    assert torch.equal(torch.isnan(a), torch.isnan(b))


def test_apply_random_mask_fraction():
    n = 10
    base = torch.tensor(np.corrcoef(_X(n=n, seed=9)))
    torch.manual_seed(0)
    out = MatrixFactorization.apply_random_mask(base.clone(), 0.5)
    off_diag_total = n * n - n
    n_masked = int(torch.isnan(out).sum().item())
    assert n_masked == int(0.5 * off_diag_total)


def test_apply_random_mask_rejects_bad_percentage():
    base = torch.eye(4)
    with pytest.raises(ValueError):
        MatrixFactorization.apply_random_mask(base, 1.5)


def test_apply_IQR_mask_sets_extremes():
    base = torch.tensor(np.corrcoef(_X(seed=10)))
    out = MatrixFactorization.apply_IQR_mask(base.clone())
    vals = out[~torch.isnan(out)].unique()
    # every non-nan value is either -2 or 2
    assert set(v.item() for v in vals).issubset({-2.0, 2.0})


# ------------------------------------------------------------- calculate_loss
def test_calculate_loss_ignores_nan_diagonal():
    n = 6
    m = torch.tensor(np.corrcoef(_X(n=n, seed=11)))
    t = torch.tensor(np.corrcoef(_X(n=n, seed=12)))
    m.fill_diagonal_(np.nan)
    t.fill_diagonal_(np.nan)
    model = MatrixFactorization(n, 3)
    loss = model.calculate_loss(m, t, torch.nn.functional.l1_loss)
    mask = ~m.isnan() & ~t.isnan()
    ref = torch.nn.functional.l1_loss(m[mask], t[mask], reduction="sum")
    assert torch.allclose(loss, ref)


def test_calculate_loss_rejects_numpy():
    model = MatrixFactorization(4, 2)
    with pytest.raises(TypeError):
        model.calculate_loss(np.eye(4), np.eye(4))


# ------------------------------------------------------------- training smoke
def test_train_MF_model_loss_decreases_and_finite():
    X = _X(n=15, seed=20)
    sim = MatrixFactorization.get_correlation_matrix(X, scaled="normal")
    torch.manual_seed(0)
    model, losses, lrs = MatrixFactorization.train_MF_model(
        n_time_series=15,
        similarity_matrix=sim,
        embedding_dim=8,
        epochs=60,
        verbose=False,
    )
    total = [l[0] for l in losses]
    assert np.isfinite(total).all()
    assert torch.isfinite(model.embeddings.weight).all()
    # pairwise loss at end < start (learning happened)
    pairwise = [l[1] for l in losses]
    assert pairwise[-1] < pairwise[0]
    assert len(lrs) == len(losses)


def test_train_MF_model_reproducible_with_seed():
    X = _X(n=12, seed=21)
    sim = MatrixFactorization.get_correlation_matrix(X, scaled="normal")
    torch.manual_seed(7)
    m1, _, _ = MatrixFactorization.train_MF_model(
        n_time_series=12, similarity_matrix=sim, embedding_dim=5,
        epochs=20, verbose=False,
    )
    torch.manual_seed(7)
    m2, _, _ = MatrixFactorization.train_MF_model(
        n_time_series=12, similarity_matrix=sim, embedding_dim=5,
        epochs=20, verbose=False,
    )
    assert torch.allclose(m1.embeddings.weight, m2.embeddings.weight)
