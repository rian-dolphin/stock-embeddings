import numpy as np
import polars as pl


class ReturnsData:
    def __init__(
        self,
        daily_returns_path="Data/returns_df_611.csv",
        extras_path="Data/historical_stocks.csv",
    ):
        self.daily_returns_path = daily_returns_path
        self.extras_path = extras_path
        self._get_data()
        self._get_extras()
        self.period = 1

    def _get_data(self):
        daily_returns_df = pl.read_csv(self.daily_returns_path)
        # -- The first column holds the dates (empty header in the CSV files).
        date_col = daily_returns_df.columns[0]
        daily_returns_df = daily_returns_df.rename({date_col: "date"}).with_columns(
            pl.col("date").str.to_datetime()
        )
        # -- Sort columns so a ticker's embedding index (its column position)
        # - always matches the sorted self.tickers list. The date column is kept
        # - first, followed by the alphabetically sorted ticker columns.
        ticker_cols = sorted(c for c in daily_returns_df.columns if c != "date")
        daily_returns_df = daily_returns_df.select(["date"] + ticker_cols)
        self.daily_returns_df = daily_returns_df
        self.returns_df = self.daily_returns_df

    def _ticker_columns(self, df):
        """Non-date columns of a returns frame (the ticker columns)."""
        return [c for c in df.columns if c != "date"]

    def _get_extras(self, misc_include=False):
        """
        Returns
        -------
        tickers : list
            Tickers in the order of index.
        ticker2idx : dict
            Dictionary to convert from ticker to index.
        idx2ticker : dict
            Dictionary to convert from index to ticker.
        sectors : list
            Sectors in the order of index.
        industries : list
            Industries in the order of index.
        names : list
            Company names in the order of index.

        """
        # -- Read in data containing sectors etc
        self.extras_df = pl.read_csv(self.extras_path)

        # -- Optionally remove the miscellaneous category
        # - Note: This is removed already in the returns_df_611.csv
        if not misc_include:
            temp_tickers = self._ticker_columns(self.returns_df)
            misc_tickers = (
                self.extras_df.filter(pl.col("sector") == "MISCELLANEOUS")["ticker"]
                .to_list()
            )
            drop_cols = [t for t in misc_tickers if t in temp_tickers]
            if drop_cols:
                # Drop from the underlying daily frame and re-point returns_df so
                # both stay consistent (the pandas version mutated one shared
                # object in place).
                self.daily_returns_df = self.daily_returns_df.drop(drop_cols)
                self.returns_df = self.daily_returns_df

        # -- Get tickers list (non-date columns)
        tickers = self._ticker_columns(self.returns_df)
        self.tickers = sorted(tickers)

        # -- Every ticker must have metadata, otherwise the sector/industry/name
        # - lists below would silently misalign with the embedding indices
        missing = set(tickers) - set(self.extras_df["ticker"].to_list())
        if missing:
            raise ValueError(
                f"Tickers missing from {self.extras_path}: {sorted(missing)}"
            )

        # -- Create dict to act as mapping between tickers and index
        # - This is useful for extracting specific embeddings from the embedding matrix
        self.ticker2idx = {ticker: idx for (idx, ticker) in enumerate(tickers)}
        self.idx2ticker = {idx: ticker for (idx, ticker) in enumerate(tickers)}

        # -- Store the sectors, industries, names
        self.sectors = [
            self.extras_df.filter(pl.col("ticker") == ticker)["sector"][0]
            for ticker in tickers
        ]
        self.industries = [
            self.extras_df.filter(pl.col("ticker") == ticker)["industry"][0]
            for ticker in tickers
        ]
        self.names = [
            self.extras_df.filter(pl.col("ticker") == ticker)["name"][0]
            for ticker in tickers
        ]

        self._validate_alignment()

    def _validate_alignment(self):
        """Embedding index i must correspond to column i of returns_df; every
        ticker-indexed attribute (tickers, ticker2idx, sectors, ...) has to
        follow that same order or downstream results silently misalign."""
        columns = self._ticker_columns(self.returns_df)
        if self.tickers != columns:
            raise ValueError(
                "self.tickers is not aligned with returns_df column order; "
                "embedding indices would not match ticker metadata."
            )
        if any(self.ticker2idx[t] != i for i, t in enumerate(columns)):
            raise ValueError(
                "ticker2idx is not aligned with returns_df column order."
            )

    def change_returns_period(self, period):
        """Allows us to change from daily to weekly etc.
        This method will leave the self.daily_returns_df intact
        The new dataframe of weekly etc. returns will be stored in self.returns_df

        Args:
            period (int): Number of days

        """
        if period == 1:
            print("No change made because period entered is 1")
            self.returns_df = self.daily_returns_df.clone()
            self.period = 1
        else:
            ticker_cols = self._ticker_columns(self.daily_returns_df)
            # -- Compound daily returns to a price series per ticker, take every
            # - `period`-th row (starting at row 0), then per-column pct change,
            # - dropping the first (undefined) row. Mirrors the pandas
            # - (1+daily).cumprod()[::period].pct_change()[1:] pipeline.
            cumprod_df = self.daily_returns_df.select(
                [pl.col("date")]
                + [(1 + pl.col(c)).cum_prod().alias(c) for c in ticker_cols]
            )
            sub_df = cumprod_df.gather_every(period)
            returns_df = sub_df.select(
                [pl.col("date")]
                + [(pl.col(c) / pl.col(c).shift(1) - 1).alias(c) for c in ticker_cols]
            ).slice(1)
            self.daily_returns_df = self.daily_returns_df.clone()
            self.returns_df = returns_df

            self.period = period

    def train_test_split(self, train_pct=0.7):
        """Adds the train_cutoff_date, train_returns_df and test_returns_df attributes.
        These are DAILY returns and will not reflect a change in returns period from change_returns_period.

        Args:
            train_pct (float, optional): Proportion of data in the training set. Defaults to 0.7.
        """
        if train_pct == 1:
            self.train_returns_df = self.returns_df
        elif train_pct < 1:
            self.train_cutoff_date = self.daily_returns_df["date"][
                int(self.daily_returns_df.height * train_pct)
            ]
            self.train_returns_df_daily = self.daily_returns_df.filter(
                pl.col("date") <= self.train_cutoff_date
            )
            self.test_returns_df_daily = self.daily_returns_df.filter(
                pl.col("date") >= self.train_cutoff_date
            ).slice(1)

            if self.period == 1:
                self.train_returns_df = self.train_returns_df_daily
                self.test_returns_df = self.test_returns_df_daily
            else:
                self.train_returns_df = self.returns_df.filter(
                    pl.col("date") <= self.train_cutoff_date
                )
                self.test_returns_df = self.returns_df.filter(
                    pl.col("date") >= self.train_cutoff_date
                ).slice(1)
        else:
            return ValueError("Train percentage must be between 0 and 1")

    def get_price_df(self, daily=False):
        source = self.daily_returns_df if daily else self.returns_df
        ticker_cols = self._ticker_columns(source)
        return source.select(
            [pl.col("date")]
            + [(1 + pl.col(c)).cum_prod().alias(c) for c in ticker_cols]
        )

    def get_train_test_tickers(
        self, train_pct: float = 0.7, seed: int = 42
    ) -> tuple[list, list]:
        np.random.seed(seed)
        train_tickers = sorted(
            np.random.choice(
                self.tickers, size=int(train_pct * len(self.tickers)), replace=False
            )
        )
        test_tickers = [
            ticker for ticker in self.tickers if ticker not in train_tickers
        ]
        return (train_tickers, test_tickers)
