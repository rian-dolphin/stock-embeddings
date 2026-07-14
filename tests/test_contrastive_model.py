"""Tests for the ported ContrastiveMultiPN model (stock_embeddings.models.contrastive)."""

import math

import numpy as np
import torch

from stock_embeddings.losses import IndividualSigmoidLoss
from stock_embeddings.models.contrastive import ContrastiveMultiPN


N_TIME_SERIES = 40
EMBED_DIM = 8
N_SECTORS = 4


def _index_samples(n=N_TIME_SERIES):
    return [
        (
            i,
            [(i + 1) % n, (i + 2) % n],
            [(i + 7) % n, (i + 13) % n, (i + 21) % n],
        )
        for i in range(n)
    ]


def _sectors(n=N_TIME_SERIES, n_sectors=N_SECTORS):
    # Balanced sectors so StratifiedKFold(5) / SMOTE inside get_sector_score work.
    return [f"SECTOR_{i % n_sectors}" for i in range(n)]


def test_instantiation():
    torch.manual_seed(0)
    criterion = IndividualSigmoidLoss()
    model = ContrastiveMultiPN(N_TIME_SERIES, EMBED_DIM, criterion)
    assert model.embeddings.weight.shape == (N_TIME_SERIES, EMBED_DIM)
    assert model.criterion is criterion
    # Loss history starts empty.
    for key in ("total", "contrastive", "positive", "negative", "regularization"):
        assert model.losses[key] == []


def test_tiny_training_run_records_finite_losses():
    torch.manual_seed(0)
    np.random.seed(0)
    criterion = IndividualSigmoidLoss()
    model = ContrastiveMultiPN(N_TIME_SERIES, EMBED_DIM, criterion)

    epochs = 2
    model.train(
        index_samples=_index_samples(),
        batch_size=8,
        learning_rate=0.01,
        epochs=epochs,
        normalize=False,
        early_stopping=False,
        eval_sectors=_sectors(),
    )

    # Losses recorded for each epoch and finite.
    for key in ("total", "contrastive", "positive", "negative"):
        assert len(model.losses[key]) == epochs
        assert all(isinstance(v, float) for v in model.losses[key])
        assert all(math.isfinite(v) for v in model.losses[key])
    assert len(model.losses["learning_rate"]) == epochs


def test_training_run_with_regularization():
    torch.manual_seed(1)
    np.random.seed(1)
    criterion = IndividualSigmoidLoss()
    model = ContrastiveMultiPN(N_TIME_SERIES, EMBED_DIM, criterion)

    epochs = 2
    model.train(
        index_samples=_index_samples(),
        batch_size=8,
        learning_rate=0.01,
        epochs=epochs,
        regularization_weight=0.1,
        normalize=False,
        early_stopping=False,
        eval_sectors=_sectors(),
    )
    assert len(model.losses["regularization"]) == epochs
    assert all(math.isfinite(v) for v in model.losses["regularization"])
    # Regularization active -> non-zero contribution recorded.
    assert any(v > 0 for v in model.losses["regularization"])
