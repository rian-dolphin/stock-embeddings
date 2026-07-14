"""Characterization tests for utils/training_helpers.py."""

import numpy as np
import pytest
import torch

from stock_embeddings.models.classification import ClassificationEmbeddings
from stock_embeddings.training import (
    get_train_loader,
    should_early_stop,
    train_embeddings_from_tgt_context_sets,
    validate_params,
)


# --------------------------------------------------------------------------- #
# validate_params
# --------------------------------------------------------------------------- #
def test_validate_params_requires_tgt_context_sets():
    with pytest.raises(ValueError, match="tgt_context_sets must be provided"):
        validate_params(model=None, embedding_dim=8, tgt_context_sets=None)


def test_validate_params_requires_model_or_dim():
    with pytest.raises(ValueError, match="embedding_dim must be specified"):
        validate_params(model=None, embedding_dim=None, tgt_context_sets=[(0, [1])])


def test_validate_params_rejects_model_and_dim():
    model = ClassificationEmbeddings(3, 2)
    with pytest.raises(ValueError, match="Do not provide embedding_dim"):
        validate_params(model=model, embedding_dim=8, tgt_context_sets=[(0, [1])])


def test_validate_params_valid_cases_return_none():
    model = ClassificationEmbeddings(3, 2)
    assert validate_params(model=model, embedding_dim=None, tgt_context_sets=[(0, [1])]) is None
    assert validate_params(model=None, embedding_dim=8, tgt_context_sets=[(0, [1])]) is None


# --------------------------------------------------------------------------- #
# get_train_loader
# --------------------------------------------------------------------------- #
def test_get_train_loader_shapes_and_dtypes():
    tcs = [(i % 4, [(i + 1) % 4, (i + 2) % 4]) for i in range(20)]
    loader = get_train_loader(tcs, batch_size=4)
    x_batch, y_batch = next(iter(loader))
    assert x_batch.shape == (4, 2)  # (batch, context_size)
    assert y_batch.shape == (4,)
    assert x_batch.dtype == torch.long
    assert y_batch.dtype == torch.long


def test_get_train_loader_drop_last():
    # 10 samples, batch 4, drop_last=True -> 2 full batches (8 samples).
    tcs = [(i % 3, [(i + 1) % 3, (i + 2) % 3]) for i in range(10)]
    loader = get_train_loader(tcs, batch_size=4)
    batches = list(loader)
    assert len(batches) == 2
    assert all(len(y) == 4 for _, y in batches)


# --------------------------------------------------------------------------- #
# should_early_stop
# --------------------------------------------------------------------------- #
def test_should_early_stop_true_when_flat_after_epoch_2():
    # epoch > 2 and |last - prev| < 0.0002
    assert should_early_stop([1.0, 0.5, 0.30000, 0.30005], epoch=3) is True


def test_should_early_stop_false_before_epoch_3():
    assert should_early_stop([1.0, 0.9, 0.8], epoch=2) is False


def test_should_early_stop_false_when_still_improving():
    assert should_early_stop([1.0, 0.5, 0.3, 0.1], epoch=3) is False


# --------------------------------------------------------------------------- #
# End-to-end smoke test
# --------------------------------------------------------------------------- #
def test_train_embeddings_smoke_loss_decreases():
    torch.manual_seed(0)
    np.random.seed(0)
    n = 6
    # Deterministic, learnable structure: target i, context = its two ring neighbors.
    tcs = []
    for _ in range(40):
        for i in range(n):
            tcs.append((i, [(i + 1) % n, (i + 2) % n]))

    model = ClassificationEmbeddings(n, 8)
    trained, losses = train_embeddings_from_tgt_context_sets(
        n_time_series=n,
        tgt_context_sets=tcs,
        model=model,
        epochs=15,
        batch_size=16,
        early_stopping=False,
        verbose=False,
    )
    assert isinstance(losses, list)
    assert all(isinstance(x, float) for x in losses)
    assert len(losses) == 15
    assert isinstance(trained, ClassificationEmbeddings)
    # Loss decreases over training (seeded -> deterministic).
    assert losses[-1] < losses[0]


def test_train_embeddings_model_none_builds_classification_model():
    # When model is None, a ClassificationEmbeddings of the requested
    # dimensions is created and trained.
    torch.manual_seed(0)
    tcs = [(i % 4, [(i + 1) % 4, (i + 2) % 4]) for i in range(20)]
    model, losses = train_embeddings_from_tgt_context_sets(
        n_time_series=4,
        tgt_context_sets=tcs,
        model=None,
        embedding_dim=8,
        epochs=1,
        batch_size=4,
        verbose=False,
    )
    assert isinstance(model, ClassificationEmbeddings)
    assert model.embeddings.weight.shape == (4, 8)
    assert len(losses) == 1 and all(isinstance(x, float) for x in losses)
