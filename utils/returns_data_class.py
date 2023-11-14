import pandas as pd
import numpy as np


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
        daily_returns_df = pd.read_csv(self.daily_returns_path, index_col=0)
        daily_returns_df.index = pd.to_datetime(daily_returns_df.index)
        self.daily_returns_df = daily_returns_df
        self.returns_df = self.daily_returns_df

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
        self.extras_df = pd.read_csv(self.extras_path)

        # -- Optionally remove the miscellaneous category
        # - Note: This is removed already in the returns_df_611.csv
        if not misc_include:
            temp_tickers = list(self.returns_df.columns)
            for ticker in self.extras_df[
                self.extras_df.sector == "MISCELLANEOUS"
            ].ticker:
                if ticker in temp_tickers:
                    self.returns_df.drop([ticker], axis=1, inplace=True)

        # -- Get tickers list
        tickers = list(self.returns_df.columns)
        self.tickers = sorted(tickers)

        # -- Create dict to act as mapping between tickers and index
        # - This is useful for extracting specific embeddings from the embedding matrix
        self.ticker2idx = {ticker: idx for (idx, ticker) in enumerate(tickers)}
        self.idx2ticker = {idx: ticker for (idx, ticker) in enumerate(tickers)}

        # -- Store the sectors, industries, names
        self.sectors = [
            self.extras_df[self.extras_df.ticker == ticker].sector.values[0]
            for ticker in tickers
        ]
        self.industries = [
            self.extras_df[self.extras_df.ticker == ticker].industry.values[0]
            for ticker in tickers
        ]
        self.names = [
            self.extras_df[self.extras_df.ticker == ticker].name.values[0]
            for ticker in tickers
        ]

    def change_returns_period(self, period):
        """Allows us to change from daily to weekly etc.
        This method will leave the self.daily_returns_df intact
        The new dataframe of weekly etc. returns will be stored in self.returns_df

        Args:
            period (int): Number of days

        """
        if period == 1:
            print("No change made because period entered is 1")
            self.returns_df = self.daily_returns_df.copy()
            self.period = 1
        else:
            returns_df = (1 + self.daily_returns_df).cumprod()[::period]
            returns_df = returns_df.pct_change()[1:]
            self.daily_returns_df = self.daily_returns_df.copy()
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
            self.train_cutoff_date = self.daily_returns_df.iloc[
                int(len(self.daily_returns_df) * train_pct)
            ].name
            self.train_returns_df_daily = self.daily_returns_df.loc[
                : self.train_cutoff_date
            ]
            self.test_returns_df_daily = self.daily_returns_df.loc[
                self.train_cutoff_date :
            ].iloc[1:]

            if self.period == 1:
                self.train_returns_df = self.train_returns_df_daily
                self.test_returns_df = self.test_returns_df_daily
            else:
                self.train_returns_df = self.returns_df.loc[: self.train_cutoff_date]
                self.test_returns_df = self.returns_df.loc[
                    self.train_cutoff_date :
                ].iloc[1:]
        else:
            return ValueError("Train percentage must be between 0 and 1")

    def get_price_df(self, daily=False):
        if daily:
            return (1 + self.daily_returns_df).cumprod(axis=0)
        else:
            return (1 + self.returns_df).cumprod(axis=0)

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
