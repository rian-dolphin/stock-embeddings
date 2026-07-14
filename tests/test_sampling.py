"""Tests for the ported sampling utilities (stock_embeddings.sampling)."""

import numpy as np
import polars as pl
import pytest
import torch
from scipy.stats import norm

from stock_embeddings.data import ReturnsData
from stock_embeddings.sampling import (
    MultiPosNegDataset,
    get_cooccurrence_counts,
    get_pairwise_p_values,
    get_sampling_distribution,
    sample_tgt_context_sets_from_distribution,
)
# Aliased: the source function name starts with "test_" and would otherwise be
# collected as a test case by pytest.
from stock_embeddings.sampling import (
    test_ticker_cooccurrence_significance as ticker_cooccurrence_significance,
)


# --------------------------------------------------------------------------- #
# MultiPosNegDataset
# --------------------------------------------------------------------------- #
def _index_samples(n=6, n_pos=2, n_neg=3):
    return [
        (i, [(i + 1) % n, (i + 2) % n], [(i + 3) % n, (i + 4) % n, (i + 5) % n])
        for i in range(n)
    ]


def test_dataset_len():
    samples = _index_samples(n=6)
    ds = MultiPosNegDataset(samples)
    assert len(ds) == 6


def test_dataset_getitem_shapes_and_dtypes():
    samples = _index_samples(n=6, n_pos=2, n_neg=3)
    ds = MultiPosNegDataset(samples)
    anchor, positive, negative = ds[0]
    assert anchor.dtype == torch.int32
    assert positive.dtype == torch.int32
    assert negative.dtype == torch.int32
    assert anchor.shape == torch.Size([])
    assert positive.shape == (2,)
    assert negative.shape == (3,)


def test_dataset_values_preserved():
    samples = [(0, [1, 2], [3, 4, 5])]
    ds = MultiPosNegDataset(samples)
    anchor, positive, negative = ds[0]
    assert int(anchor) == 0
    assert positive.tolist() == [1, 2]
    assert negative.tolist() == [3, 4, 5]


# --------------------------------------------------------------------------- #
# get_cooccurrence_counts
# --------------------------------------------------------------------------- #
def test_get_cooccurrence_counts_structure(make_returns_dataset):
    rpath, epath, _, _ = make_returns_dataset(include_misc=False)
    data = ReturnsData(rpath, epath)  # tickers: AAA, BBB, CCC, DDD
    idx = data.ticker2idx
    # Uniform context length (the source flattens per-target contexts into an array).
    # AAA co-occurs with BBB (twice) and CCC (twice); DDD never -> zero-filled.
    tgt_context_sets = [
        (idx["AAA"], [idx["BBB"], idx["CCC"]]),
        (idx["AAA"], [idx["BBB"], idx["CCC"]]),
        (idx["BBB"], [idx["AAA"], idx["CCC"]]),
    ]
    dist = get_cooccurrence_counts(tgt_context_sets, data, verbose=False)
    # One entry per ticker.
    assert set(dist.keys()) == set(data.tickers)
    # AAA counts: BBB=2, CCC=2, and DDD filled with 0 (self excluded).
    assert dist["AAA"]["BBB"] == 2
    assert dist["AAA"]["CCC"] == 2
    assert dist["AAA"]["DDD"] == 0
    assert "AAA" not in dist["AAA"]  # self excluded


# --------------------------------------------------------------------------- #
# test_ticker_cooccurrence_significance
# --------------------------------------------------------------------------- #
def _cooccurrence_df():
    # First column "ticker" = row (sample) tickers; remaining columns = query
    # tickers. Matches the polars matrix layout used by get_pairwise_p_values.
    return pl.DataFrame(
        {
            "ticker": ["AAA", "BBB", "CCC"],
            "AAA": [0, 8, 2],
            "BBB": [5, 0, 5],
            "CCC": [3, 3, 0],
        }
    )


def test_significance_matches_manual_positive():
    df = _cooccurrence_df()
    p = ticker_cooccurrence_significance(
        "AAA", "BBB", df, test_direction="positive_samples"
    )
    observed = df.filter(pl.col("ticker") == "BBB")["AAA"][0]
    total = df["AAA"].sum()
    exp = 1 / df.height
    obs_prop = observed / total
    ts = (obs_prop - exp) / np.sqrt(exp * (1 - exp) / total)
    assert p == pytest.approx(norm.sf(ts))


def test_significance_alpha_returns_bool():
    df = _cooccurrence_df()
    result = ticker_cooccurrence_significance(
        "AAA", "BBB", df, alpha=0.05, test_direction="positive_samples"
    )
    assert isinstance(result, (bool, np.bool_))


# --------------------------------------------------------------------------- #
# get_pairwise_p_values
# --------------------------------------------------------------------------- #
def test_get_pairwise_p_values_columns(make_returns_dataset):
    rpath, epath, _, _ = make_returns_dataset(include_misc=False)
    data = ReturnsData(rpath, epath)
    idx = data.ticker2idx
    tgt_context_sets = [
        (idx["AAA"], [idx["BBB"], idx["CCC"]]),
        (idx["AAA"], [idx["BBB"], idx["DDD"]]),
        (idx["BBB"], [idx["AAA"], idx["CCC"]]),
        (idx["CCC"], [idx["AAA"], idx["DDD"]]),
    ]
    dist = get_cooccurrence_counts(tgt_context_sets, data, verbose=False)
    df = get_pairwise_p_values(dist, data, return_dataframe=True)
    assert df.columns == [
        "query_ticker",
        "sample_ticker",
        "count",
        "p_value",
        "query_ticker_idx",
        "sample_ticker_idx",
    ]
    assert df.height > 0
    # idx columns are the ticker2idx mapping of the corresponding ticker columns.
    assert df["query_ticker_idx"].to_list() == [
        data.ticker2idx[t] for t in df["query_ticker"].to_list()
    ]
    assert df["sample_ticker_idx"].to_list() == [
        data.ticker2idx[t] for t in df["sample_ticker"].to_list()
    ]


# --------------------------------------------------------------------------- #
# get_sampling_distribution
# --------------------------------------------------------------------------- #
def _pairwise_df():
    return pl.DataFrame(
        {
            "query_ticker": ["AAA", "AAA", "AAA", "AAA"],
            "sample_ticker": ["BBB", "CCC", "DDD", "EEE"],
            "count": [10, 6, 4, 1],
            "p_value": [0.01, 0.02, 0.05, 0.9],
        }
    )


def test_get_sampling_distribution_positive_structure():
    df = _pairwise_df()
    dist = get_sampling_distribution(
        "AAA", df, sample_type="positive_samples", positive_threshold=0.1
    )
    # 3 samples pass p_value < 0.1 (BBB, CCC, DDD); EEE excluded.
    assert dist.shape == (3, 3)
    assert set(dist[:, 1]) == {"BBB", "CCC", "DDD"}
    probs = dist[:, 2].astype(float)
    assert probs.sum() == pytest.approx(1.0)
    # counts 10,6,4 -> proportional probabilities
    assert np.allclose(probs, np.array([10, 6, 4]) / 20)


def test_get_sampling_distribution_invalid_type_raises():
    df = _pairwise_df()
    with pytest.raises(ValueError, match="Invalid sample_type"):
        get_sampling_distribution("AAA", df, sample_type="bogus")


# --------------------------------------------------------------------------- #
# sample_tgt_context_sets_from_distribution
# --------------------------------------------------------------------------- #
def test_sample_from_distribution_structure():
    np.random.seed(0)
    filtered = np.array(
        [
            ["AAA", "BBB", 0.5],
            ["AAA", "CCC", 0.3],
            ["AAA", "DDD", 0.2],
        ],
        dtype=object,
    )
    result = sample_tgt_context_sets_from_distribution(filtered, n_samples=4, sample_size=2)
    assert len(result) == 4
    for query, samples in result:
        assert query == "AAA"
        assert len(samples) == 2
        assert len(set(samples)) == 2  # sampled without replacement
        assert all(s in {"BBB", "CCC", "DDD"} for s in samples)


def test_sample_from_distribution_multiple_queries_raises():
    filtered = np.array(
        [["AAA", "BBB", 0.5], ["ZZZ", "CCC", 0.5]], dtype=object
    )
    with pytest.raises(ValueError, match="single distribution"):
        sample_tgt_context_sets_from_distribution(filtered, n_samples=1, sample_size=1)


def test_sample_from_distribution_insufficient_elements_raises():
    filtered = np.array([["AAA", "BBB", 1.0]], dtype=object)
    with pytest.raises(ValueError, match="Not enough elements"):
        sample_tgt_context_sets_from_distribution(filtered, n_samples=1, sample_size=2)
