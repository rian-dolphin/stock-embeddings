"""Tests for the ported contrastive loss classes (stock_embeddings.losses)."""

import pytest
import torch
import torch.nn.functional as F

from stock_embeddings.losses import (
    AggPos_IndNeg,
    AggregateSigmoidLoss,
    BaseContrastiveLoss,
    IndividualSigmoidLoss,
    IndPos_AggNeg,
    IndPos_AggSoftmax,
    JointPos_MarginalNeg,
)

BATCH, EMBED, N_POS, N_NEG = 4, 3, 2, 3

ALL_LOSSES = [
    IndividualSigmoidLoss,
    AggregateSigmoidLoss,
    IndPos_AggSoftmax,
    AggPos_IndNeg,
    IndPos_AggNeg,
    JointPos_MarginalNeg,
]


def _inputs(seed=0):
    torch.manual_seed(seed)
    anchor = torch.randn(BATCH, EMBED)
    positive = torch.randn(BATCH, N_POS, EMBED)
    negative = torch.randn(BATCH, N_NEG, EMBED)
    return anchor, positive, negative


def _is_finite_scalar(t):
    return isinstance(t, torch.Tensor) and t.dim() == 0 and torch.isfinite(t)


# --------------------------------------------------------------------------- #
# Base class
# --------------------------------------------------------------------------- #
def test_base_forward_not_implemented():
    loss = BaseContrastiveLoss()
    a, p, n = _inputs()
    with pytest.raises(NotImplementedError):
        loss(a, p, n)


def test_base_default_weights():
    loss = BaseContrastiveLoss()
    assert loss.positive_weight == 1
    assert loss.negative_weight == 1


def test_update_loss_weights_rejects_nonunit_ratio():
    loss = BaseContrastiveLoss()
    with pytest.raises(NotImplementedError):
        loss.update_loss_weights(1.0, 1.0, target_ratio=2)


def test_update_loss_weights_math():
    loss = BaseContrastiveLoss(positive_weight=1.0, negative_weight=1.0)
    positive_loss, negative_loss, target_magnitude = 4.0, 2.0, 5.0
    loss.update_loss_weights(positive_loss, negative_loss, target_magnitude=target_magnitude)
    loss_ratio = positive_loss / negative_loss
    magnitude_factor = target_magnitude / (2 * negative_loss)
    assert loss.positive_weight == pytest.approx(1.0 * (1 / loss_ratio) * magnitude_factor)
    assert loss.negative_weight == pytest.approx(1.0 * magnitude_factor)


# --------------------------------------------------------------------------- #
# Structure: every loss returns a (total, pos, neg) triple of finite scalars
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("loss_cls", ALL_LOSSES)
def test_forward_returns_triple_of_finite_scalars(loss_cls):
    loss = loss_cls()
    a, p, n = _inputs(seed=1)
    total, pos, neg = loss(a, p, n)
    assert _is_finite_scalar(total)
    assert _is_finite_scalar(pos)
    assert _is_finite_scalar(neg)
    # total is exactly positive + negative for all these losses
    assert torch.allclose(total, pos + neg)


# --------------------------------------------------------------------------- #
# Manual references for the two simple sigmoid losses
# --------------------------------------------------------------------------- #
def test_individual_sigmoid_matches_manual():
    loss = IndividualSigmoidLoss()
    a, p, n = _inputs(seed=2)
    total, pos, neg = loss(a, p, n)

    pos_scores = torch.einsum("bpd,bd->bp", p, a)
    neg_scores = torch.einsum("bnd,bd->bn", n, a)
    ref_pos = F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores))
    ref_neg = F.binary_cross_entropy_with_logits(neg_scores, torch.zeros_like(neg_scores))
    assert torch.allclose(pos, ref_pos)
    assert torch.allclose(neg, ref_neg)
    assert torch.allclose(total, ref_pos + ref_neg)


def test_aggregate_sigmoid_matches_manual():
    loss = AggregateSigmoidLoss()
    a, p, n = _inputs(seed=3)
    total, pos, neg = loss(a, p, n)

    pos_scores = torch.einsum("bd,bd->b", torch.mean(p, dim=1), a)
    neg_scores = torch.einsum("bd,bd->b", torch.mean(n, dim=1), a)
    ref_pos = F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores))
    ref_neg = F.binary_cross_entropy_with_logits(neg_scores, torch.zeros_like(neg_scores))
    assert torch.allclose(pos, ref_pos)
    assert torch.allclose(neg, ref_neg)


# --------------------------------------------------------------------------- #
# Weight influence
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("loss_cls", ALL_LOSSES)
def test_positive_weight_scales_positive_loss(loss_cls):
    a, p, n = _inputs(seed=4)
    _, base_pos, _ = loss_cls(positive_weight=1)(a, p, n)
    _, scaled_pos, _ = loss_cls(positive_weight=2)(a, p, n)
    assert torch.allclose(scaled_pos, 2 * base_pos)


@pytest.mark.parametrize("loss_cls", ALL_LOSSES)
def test_negative_weight_scales_negative_loss(loss_cls):
    a, p, n = _inputs(seed=5)
    _, _, base_neg = loss_cls(negative_weight=1)(a, p, n)
    _, _, scaled_neg = loss_cls(negative_weight=3)(a, p, n)
    assert torch.allclose(scaled_neg, 3 * base_neg)
