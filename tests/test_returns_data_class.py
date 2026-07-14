"""Characterization tests for utils/returns_data_class.py (ReturnsData)."""

import numpy as np
import pandas as pd

from stock_embeddings.data import ReturnsData


def test_misc_sector_ticker_removed(make_returns_dataset):
    rpath, epath, _, _ = make_returns_dataset(include_misc=True)
    rd = ReturnsData(rpath, epath)
    # DDD (MISCELLANEOUS) dropped from returns; only 3 tickers remain.
    assert "DDD" not in rd.returns_df.columns
    assert rd.tickers == ["AAA", "BBB", "CCC"]


def test_misc_kept_when_not_miscellaneous(make_returns_dataset):
    rpath, epath, _, _ = make_returns_dataset(include_misc=False)
    rd = ReturnsData(rpath, epath)
    assert "DDD" in rd.returns_df.columns
    assert rd.tickers == ["AAA", "BBB", "CCC", "DDD"]


def test_ticker_index_mappings_are_inverse(make_returns_dataset):
    rpath, epath, _, _ = make_returns_dataset(include_misc=True)
    rd = ReturnsData(rpath, epath)
    # ticker2idx/idx2ticker follow the (post-drop) column order.
    assert set(rd.ticker2idx.keys()) == set(rd.tickers)
    for ticker, idx in rd.ticker2idx.items():
        assert rd.idx2ticker[idx] == ticker


def test_sectors_industries_names_aligned_to_ticker2idx(make_returns_dataset):
    rpath, epath, _, extras_df = make_returns_dataset(include_misc=True)
    rd = ReturnsData(rpath, epath)
    # sectors/industries/names are aligned to the ticker2idx ordering, not the
    # sorted self.tickers list. Verify against the source extras frame.
    for ticker, idx in rd.ticker2idx.items():
        row = extras_df[extras_df.ticker == ticker].iloc[0]
        assert rd.sectors[idx] == row.sector
        assert rd.industries[idx] == row.industry
        assert rd.names[idx] == row["name"]


def test_change_returns_period_period_1_noop(make_returns_dataset, capsys):
    rpath, epath, _, _ = make_returns_dataset()
    rd = ReturnsData(rpath, epath)
    original = rd.returns_df.copy()
    rd.change_returns_period(1)
    out = capsys.readouterr().out
    assert "No change made" in out
    assert rd.period == 1
    pd.testing.assert_frame_equal(rd.returns_df, original)


def test_change_returns_period_compounding_math(make_returns_dataset):
    rpath, epath, _, _ = make_returns_dataset(n_days=12)
    rd = ReturnsData(rpath, epath)
    daily = rd.daily_returns_df.copy()
    period = 3
    rd.change_returns_period(period)
    # Reference: compound daily to a price series, subsample every `period` rows,
    # then take pct_change and drop the first (NaN) row.
    ref = (1 + daily).cumprod()[::period].pct_change()[1:]
    assert rd.period == period
    assert list(rd.returns_df.columns) == list(ref.columns)
    assert np.allclose(rd.returns_df.values, ref.values)


def test_train_test_split_cutoff_070(make_returns_dataset):
    rpath, epath, _, _ = make_returns_dataset(n_days=10)
    rd = ReturnsData(rpath, epath)
    rd.train_test_split(train_pct=0.7)
    # cutoff index = int(10 * 0.7) = 7 -> the 8th row's date.
    expected_cutoff = rd.daily_returns_df.iloc[7].name
    assert rd.train_cutoff_date == expected_cutoff
    # train includes rows up to and including cutoff (8 rows); test excludes it (2).
    assert len(rd.train_returns_df) == 8
    assert len(rd.test_returns_df) == 2
    # No overlap: test starts strictly after the cutoff date.
    assert rd.test_returns_df.index.min() > rd.train_cutoff_date


def test_train_test_split_full_train(make_returns_dataset):
    rpath, epath, _, _ = make_returns_dataset(n_days=10)
    rd = ReturnsData(rpath, epath)
    rd.train_test_split(train_pct=1)
    pd.testing.assert_frame_equal(rd.train_returns_df, rd.returns_df)
    # train_pct=1 does not create a test set attribute.
    assert not hasattr(rd, "test_returns_df")


def test_get_price_df_is_cumprod(make_returns_dataset):
    rpath, epath, _, _ = make_returns_dataset()
    rd = ReturnsData(rpath, epath)
    price = rd.get_price_df()
    ref = (1 + rd.returns_df).cumprod(axis=0)
    assert np.allclose(price.values, ref.values)
    price_daily = rd.get_price_df(daily=True)
    ref_daily = (1 + rd.daily_returns_df).cumprod(axis=0)
    assert np.allclose(price_daily.values, ref_daily.values)


def test_get_train_test_tickers_deterministic_and_partition(make_returns_dataset):
    rpath, epath, _, _ = make_returns_dataset(include_misc=True)
    rd = ReturnsData(rpath, epath)
    a_train, a_test = rd.get_train_test_tickers(train_pct=0.7, seed=42)
    b_train, b_test = rd.get_train_test_tickers(train_pct=0.7, seed=42)
    assert a_train == b_train and a_test == b_test  # deterministic under seed
    # partition covers all tickers with no overlap
    assert sorted(list(a_train) + list(a_test)) == sorted(rd.tickers)
    assert set(a_train).isdisjoint(a_test)
    assert len(a_train) == int(0.7 * len(rd.tickers))


def test_get_train_test_tickers_different_seed_may_differ(make_returns_dataset):
    # Not guaranteed different in general, but with enough tickers a different seed
    # gives a different split here. Use the full 4-ticker set (misc excluded keeps 3).
    rpath, epath, _, _ = make_returns_dataset(include_misc=False)
    rd = ReturnsData(rpath, epath)
    s1 = rd.get_train_test_tickers(train_pct=0.5, seed=1)
    s2 = rd.get_train_test_tickers(train_pct=0.5, seed=1)
    assert s1 == s2  # reseeded each call -> stable
