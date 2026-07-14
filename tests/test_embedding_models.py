"""Characterization tests for models/base_model.py + models/embedding_models.py."""

import numpy as np
import pytest
import torch

from stock_embeddings.models.base import BaseModel
from stock_embeddings.models.classification import ClassificationEmbeddings


def _model(n=6, dim=4, seed=0):
    torch.manual_seed(seed)
    return ClassificationEmbeddings(n, dim)


def test_forward_output_shape():
    n, dim, batch, ctx = 6, 4, 3, 2
    model = _model(n, dim)
    inputs = torch.tensor([[0, 1], [2, 3], [4, 5]], dtype=torch.long)
    out = model(inputs)
    assert out.shape == (batch, n)


def test_forward_is_log_softmax_rows_sum_to_one():
    model = _model()
    inputs = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
    out = model(inputs)
    probs = out.exp().sum(dim=1)
    assert torch.allclose(probs, torch.ones_like(probs), atol=1e-5)


def test_forward_matches_manual_reference():
    # Reference: context_emb = W[inputs]; hidden = mean over context;
    # out[b, n] = sum_d W[n, d] * hidden[b, d]; then log_softmax over n.
    model = _model(n=5, dim=3, seed=1)
    W = model.embeddings.weight.detach().numpy()
    inputs = torch.tensor([[0, 2], [1, 4]], dtype=torch.long)
    out = model(inputs).detach().numpy()

    ref = []
    for row in inputs.numpy():
        hidden = W[row].mean(axis=0)
        logits = W @ hidden
        logits = logits - logits.max()  # stable log-softmax
        log_probs = logits - np.log(np.exp(logits).sum())
        ref.append(log_probs)
    ref = np.array(ref)
    assert np.allclose(out, ref, atol=1e-5)


def test_save_load_model_roundtrip(tmp_path):
    model = _model(n=7, dim=5, seed=2)
    path = tmp_path / "model.pt"
    model.save_model(str(path))
    loaded = ClassificationEmbeddings.load_model(str(path))
    assert loaded.embeddings.weight.shape == model.embeddings.weight.shape
    assert torch.allclose(loaded.embeddings.weight, model.embeddings.weight)


def test_load_embeddings_from_numpy(capsys):
    model = ClassificationEmbeddings(3, 2)
    arr = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    model.load_embeddings_from_numpy(arr)
    assert torch.allclose(model.embeddings.weight, torch.from_numpy(arr))
    out = capsys.readouterr().out
    assert "Number of Time Series: 3" in out
    assert "Embedding Dimension: 2" in out


def test_save_and_load_embeddings_csv_roundtrip(tmp_path):
    model = ClassificationEmbeddings(4, 3)
    arr = np.arange(12, dtype=np.float64).reshape(4, 3)
    model.load_embeddings_from_numpy(arr)
    csv_path = tmp_path / "emb.csv"
    model.save_embeddings_to_csv(str(csv_path))

    model2 = ClassificationEmbeddings(4, 3)
    model2.load_embeddings_from_csv(str(csv_path))
    assert np.allclose(
        model2.embeddings.weight.detach().numpy(),
        model.embeddings.weight.detach().numpy(),
    )


def test_save_embeddings_requires_csv_extension():
    model = ClassificationEmbeddings(2, 2)
    model.load_embeddings_from_numpy(np.zeros((2, 2)))
    with pytest.raises(ValueError, match="include .csv"):
        model.save_embeddings_to_csv("bad.txt")


def test_base_model_forward_not_implemented():
    # BaseModel is abstract-in-spirit; forward raises. Note: its __init__ only
    # accepts `device`, so BaseModel cannot be constructed with (n, dim).
    bm = BaseModel()
    with pytest.raises(NotImplementedError):
        bm.forward(torch.zeros(1))
