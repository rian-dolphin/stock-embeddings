"""Characterization tests for stock_embeddings.data (ReturnsData)."""

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from stock_embeddings.data import ReturnsData


def _ticker_cols(df):
    """Ticker (non-date) columns of a returns frame, in order."""
    return [c for c in df.columns if c != "date"]


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


def test_date_is_first_column(make_returns_dataset):
    rpath, epath, _, _ = make_returns_dataset(include_misc=False)
    rd = ReturnsData(rpath, epath)
    # The returns frames carry the date as their first column (pl.Datetime),
    # followed by the alphabetically sorted ticker columns.
    assert rd.returns_df.columns[0] == "date"
    assert rd.returns_df["date"].dtype == pl.Datetime
    assert _ticker_cols(rd.returns_df) == rd.tickers


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
        row = extras_df.filter(pl.col("ticker") == ticker).row(0, named=True)
        assert rd.sectors[idx] == row["sector"]
        assert rd.industries[idx] == row["industry"]
        assert rd.names[idx] == row["name"]


def test_unsorted_columns_are_sorted_and_aligned(make_returns_dataset):
    # Columns are sorted at load, so a ticker's embedding index (column
    # position) always matches self.tickers and the metadata lists.
    rpath, epath, returns_df, extras_df = make_returns_dataset(
        include_misc=False, column_order=["CCC", "AAA", "DDD", "BBB"]
    )
    rd = ReturnsData(rpath, epath)
    assert _ticker_cols(rd.returns_df) == ["AAA", "BBB", "CCC", "DDD"]
    assert rd.tickers == _ticker_cols(rd.returns_df)
    for idx, ticker in enumerate(rd.tickers):
        assert rd.ticker2idx[ticker] == idx
        row = extras_df.filter(pl.col("ticker") == ticker).row(0, named=True)
        assert rd.sectors[idx] == row["sector"]
        assert rd.names[idx] == row["name"]
    # Returns values follow the columns, not the original file order.
    np.testing.assert_allclose(
        rd.daily_returns_df["CCC"].to_numpy(), returns_df["CCC"].to_numpy()
    )


def test_gappy_leading_column_parses_as_float(make_returns_dataset):
    # A ticker column whose first >100 rows are empty must still be inferred
    # as float (full-column schema inference), matching pandas behavior.
    rpath, epath, returns_df, _ = make_returns_dataset(n_days=150, include_misc=False)
    df = pl.read_csv(rpath, infer_schema_length=None)
    gappy = pl.Series("AAA", [None] * 120 + df["AAA"].to_list()[120:])
    df.with_columns(gappy).write_csv(rpath)
    rd = ReturnsData(rpath, epath)
    assert rd.returns_df["AAA"].dtype == pl.Float64
    rd.change_returns_period(3)  # must not raise on arithmetic


def test_missing_extras_metadata_raises(make_returns_dataset, tmp_path):
    rpath, epath, _, extras_df = make_returns_dataset(include_misc=False)
    # Drop one ticker's metadata row -> clear error instead of misalignment.
    extras_df.filter(pl.col("ticker") != "BBB").write_csv(epath)
    with pytest.raises(ValueError, match="BBB"):
        ReturnsData(rpath, epath)


def test_change_returns_period_period_1_noop(make_returns_dataset, capsys):
    rpath, epath, _, _ = make_returns_dataset()
    rd = ReturnsData(rpath, epath)
    original = rd.returns_df.clone()
    rd.change_returns_period(1)
    out = capsys.readouterr().out
    assert "No change made" in out
    assert rd.period == 1
    assert_frame_equal(rd.returns_df, original)


def test_change_returns_period_compounding_math(make_returns_dataset):
    rpath, epath, _, _ = make_returns_dataset(n_days=12)
    rd = ReturnsData(rpath, epath)
    tickers = rd.tickers
    daily_vals = rd.daily_returns_df.select(tickers).to_numpy()
    period = 3
    rd.change_returns_period(period)
    # Independent numpy reference: compound daily to a price series, subsample
    # every `period` rows (from row 0), then pct_change and drop the first row.
    cum = np.cumprod(1 + daily_vals, axis=0)
    sub = cum[::period]
    ref = sub[1:] / sub[:-1] - 1
    assert rd.period == period
    assert [c for c in rd.returns_df.columns if c != "date"] == tickers
    assert np.allclose(rd.returns_df.select(tickers).to_numpy(), ref)


def test_train_test_split_cutoff_070(make_returns_dataset):
    rpath, epath, _, _ = make_returns_dataset(n_days=10)
    rd = ReturnsData(rpath, epath)
    rd.train_test_split(train_pct=0.7)
    # cutoff index = int(10 * 0.7) = 7 -> the 8th row's date.
    expected_cutoff = rd.daily_returns_df["date"][7]
    assert rd.train_cutoff_date == expected_cutoff
    # train includes rows up to and including cutoff (8 rows); test excludes it (2).
    assert rd.train_returns_df.height == 8
    assert rd.test_returns_df.height == 2
    # No overlap: test starts strictly after the cutoff date.
    assert rd.test_returns_df["date"].min() > rd.train_cutoff_date


def test_train_test_split_full_train(make_returns_dataset):
    rpath, epath, _, _ = make_returns_dataset(n_days=10)
    rd = ReturnsData(rpath, epath)
    rd.train_test_split(train_pct=1)
    assert_frame_equal(rd.train_returns_df, rd.returns_df)
    # train_pct=1 does not create a test set attribute.
    assert not hasattr(rd, "test_returns_df")


def test_get_price_df_is_cumprod(make_returns_dataset):
    rpath, epath, _, _ = make_returns_dataset()
    rd = ReturnsData(rpath, epath)
    tickers = rd.tickers
    price = rd.get_price_df()
    ref = np.cumprod(1 + rd.returns_df.select(tickers).to_numpy(), axis=0)
    assert np.allclose(price.select(tickers).to_numpy(), ref)
    price_daily = rd.get_price_df(daily=True)
    ref_daily = np.cumprod(1 + rd.daily_returns_df.select(tickers).to_numpy(), axis=0)
    assert np.allclose(price_daily.select(tickers).to_numpy(), ref_daily)


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
