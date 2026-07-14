"""Shared fixtures for the characterization test suite.

These tests pin down the CURRENT behavior of the flat ``utils``/``models``
modules (imported via ``pythonpath = ["."]``). They are intentionally written
against observable behavior / independent reference computations so the same
assertions can be re-run against ``src/stock_embeddings`` after the restructure.
"""

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest


@pytest.fixture
def make_returns_dataset(tmp_path):
    """Factory that writes a synthetic returns CSV + extras CSV to tmp_path.

    Returns a callable producing ``(returns_csv_path, extras_csv_path,
    returns_df, extras_df)`` so individual tests can compare against the exact
    frames that were written to disk.
    """

    def _make(n_days=10, seed=0, include_misc=True, column_order=None):
        dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(n_days)]
        tickers = column_order or ["AAA", "BBB", "CCC", "DDD"]
        rng = np.random.default_rng(seed)
        # Small returns, no NaNs (keeps pct_change/cumprod math clean).
        data = rng.normal(0.0, 0.01, size=(n_days, len(tickers)))
        # First column has an empty header to match the real returns_df_611.csv
        # layout (date index written with a blank index name).
        returns_df = pl.DataFrame(
            {"": dates, **{t: data[:, j] for j, t in enumerate(tickers)}}
        )

        extras_rows = [
            ("AAA", "NYSE", "Alpha Corp", "TECHNOLOGY", "SOFTWARE"),
            ("BBB", "NYSE", "Beta Corp", "FINANCE", "MAJOR BANKS"),
            ("CCC", "NASDAQ", "Gamma Corp", "TECHNOLOGY", "SEMICONDUCTORS"),
        ]
        if include_misc:
            # DDD is in a MISCELLANEOUS sector -> should be dropped by ReturnsData.
            extras_rows.append(
                ("DDD", "NYSE", "Delta Corp", "MISCELLANEOUS", "N/A")
            )
        else:
            extras_rows.append(("DDD", "NYSE", "Delta Corp", "ENERGY", "OIL"))

        extras_df = pl.DataFrame(
            extras_rows,
            schema=["ticker", "exchange", "name", "sector", "industry"],
            orient="row",
        )

        returns_path = tmp_path / "returns.csv"
        extras_path = tmp_path / "extras.csv"
        returns_df.write_csv(returns_path)
        extras_df.write_csv(extras_path)

        return str(returns_path), str(extras_path), returns_df, extras_df

    return _make
