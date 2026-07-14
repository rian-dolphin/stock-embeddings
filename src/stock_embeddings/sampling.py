from typing import Literal

import numpy as np
import polars as pl
import torch
from scipy.stats import norm
from torch.utils.data import Dataset
from tqdm import tqdm

from stock_embeddings.data import ReturnsData


class MultiPosNegDataset(Dataset):
    def __init__(self, index_samples):
        self.anchors = torch.tensor([xi[0] for xi in index_samples]).int()
        self.positive_sets = torch.tensor([xi[1] for xi in index_samples]).int()
        self.negative_sets = torch.tensor([xi[2] for xi in index_samples]).int()

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, idx):
        return self.anchors[idx], self.positive_sets[idx], self.negative_sets[idx]


def get_cooccurrence_counts(
    tgt_context_sets: list, data: ReturnsData, verbose: bool = True
) -> dict:
    """
    Calculates the cooccurrence counts of stock tickers within target-context sets derived from financial data.

    This function iterates over all stock tickers in the provided financial data, computing the cooccurrence
    counts for each ticker based on the input target-context sets. The cooccurrences are counted for each ticker
    with every other ticker in the data, including setting counts to zero for tickers that do not cooccur.

    Parameters:
    - tgt_context_sets (list): A list of tuples, each containing a stock ticker index and a list of indices
      representing the context (or cooccurring) stock tickers.
    - data ('ReturnsData'): An instance of the ReturnsData class, which must contain 'tickers', 'ticker2idx', and
      'idx2ticker' attributes. 'tickers' is a list of all ticker symbols, 'ticker2idx' is a dictionary mapping
      ticker symbols to their corresponding indices, and 'idx2ticker' is a dictionary mapping indices back to
      ticker symbols.

    Returns:
    - dict: A dictionary where each key is a stock ticker symbol and each value is another dictionary mapping
      cooccurring ticker symbols to their respective cooccurrence counts.

    Example usage:
    - positive_sample_distributions = get_cooccurrence_counts(positive_tgt_context_sets, data)

    """
    distributions = {}
    for ticker in tqdm(data.tickers, disable=not verbose):
        i = data.ticker2idx[ticker]
        all_samples = np.array(
            [xi[1] for xi in tgt_context_sets if xi[0] == i]
        ).flatten()
        counts = pl.Series("value", all_samples).value_counts()
        sample_count = {
            data.idx2ticker[int(v)]: int(c)
            for v, c in zip(
                counts["value"].to_list(), counts["count"].to_list()
            )
        }
        # Add zero cooccurrences
        zero_cooccurrences = (
            set(data.tickers) - set(sample_count.keys()) - set([ticker])
        )
        sample_count.update(dict.fromkeys(list(zero_cooccurrences), 0))

        distributions[ticker] = sample_count
    return distributions


def test_ticker_cooccurrence_significance(
    t1: str,
    t2: str,
    distributions_df: dict,
    test_direction: Literal[
        "positive_samples", "negative_samples"
    ] = "positive_samples",
    alpha: float | None = None,
    verbose: bool = False,
) -> float | bool:
    """
    Tests the significance of the cooccurrence of two stock tickers within given distributions,
    returning either the p-value or a boolean indicating significance based on a specified alpha level.

    This function calculates the observed proportion of cooccurrences for two specified stock tickers and compares it
    against an expected proportion under the assumption of equal frequency distribution. A statistical test (Z-test)
    is then performed to determine the significance of the observed proportion, taking into account the specified
    direction of interest (positive or negative samples).

    Parameters:
    - t1 (str): The first stock ticker symbol.
    - t2 (str): The second stock ticker symbol, cooccurrence with t1 is tested.
    - distributions_df (pl.DataFrame): A polars dataframe containing the cooccurrence
      distributions of stock tickers. The first column ``ticker`` holds the row
      (sample) ticker labels; the remaining columns are one per query ticker.
    - test_direction (Literal["positive_samples", "negative_samples"], optional): Specifies the direction of the test;
      'positive_samples' for testing higher than expected cooccurrences, 'negative_samples' for lower.
      Defaults to 'positive_samples'.
    - alpha (float | None, optional): The significance level for determining whether the observed cooccurrence is
      statistically significant. If None, the function returns the p-value. Otherwise, it returns a boolean indicating
      whether the p-value is less than alpha. Defaults to None.
    - verbose (bool, optional): If True, prints detailed test results including observed proportion, expected proportion,
      test statistic, and p-value. Defaults to False.

    Returns:
    - Union[float, bool]: Depending on the value of alpha, either returns the p-value of the significance test
      (if alpha is None) or a boolean indicating whether the cooccurrence is statistically significant at the
      specified alpha level.

    The function is useful for analyzing relationships between stock tickers in financial data, helping to identify
    pairs of tickers that cooccur more or less frequently than would be expected by chance alone based on their proportions.

    Example usage: test_ticker_cooccurrence_significance("JPM", "C", positive_sample_distributions, verbose=True, test_direction="positive_samples")
    """

    # Cast to numpy floats so a zero column total yields nan (via numpy
    # division) rather than a Python ZeroDivisionError, matching the original
    # pandas/numpy behavior.
    observed_count = np.float64(distributions_df.filter(pl.col("ticker") == t2)[t1][0])
    t1_samples_total = np.float64(distributions_df[t1].sum())
    expected_proportion = 1 / distributions_df.height
    observed_proportion = observed_count / t1_samples_total
    # Test statistic - see https://online.stat.psu.edu/statprogram/reviews/statistical-concepts/proportions
    test_statistic = (observed_proportion - expected_proportion) / (
        np.sqrt(expected_proportion * (1 - expected_proportion) / t1_samples_total)
    )

    # p_value = norm.sf(abs(z_score))  # two-tailed test
    if test_direction == "positive_samples":
        p_value = norm.sf(test_statistic)  # one-tailed test
    elif test_direction == "negative_samples":
        p_value = norm.cdf(test_statistic)

    if verbose:
        print(f"Observed Count: {observed_proportion}")
        print(f"Expected Count: {expected_proportion}")
        print(f"Test Statistic: {test_statistic}")
        print(f"P-value: {p_value}")
    if alpha is None:
        return p_value
    else:
        return p_value < alpha


def get_pairwise_p_values(
    positive_sample_distributions: dict,
    data: ReturnsData,
    return_dataframe: bool = True,
) -> pl.DataFrame | list:
    """
    Calculates the p-values for the significance of cooccurrence between pairs of stock tickers
    based on positive sample distributions.

    This function iterates over each ticker and its cooccurring tickers within the positive sample distributions,
    computing the p-value that tests the significance of their cooccurrence. It leverages the
    `test_ticker_cooccurrence_significance` function to perform statistical tests for each pair, determining whether
    the observed cooccurrence count is significantly higher than expected under a model of random distribution.

    Parameters:
    - positive_sample_distributions (dict): A dictionary where each key is a stock ticker symbol and each value is
      another dictionary mapping cooccurring ticker symbols to their respective cooccurrence counts, as generated
      by `get_cooccurrence_counts`.
    - data: An instance of the ReturnsData class, or similar, containing at least 'tickers', 'ticker2idx', and
      'idx2ticker' attributes for mapping between ticker symbols and indices.
    - return_dataframe (bool, optional): If True, returns the results as a polars DataFrame. Otherwise, returns a list
      of tuples. Defaults to True.

    Returns:
    - Union[pl.DataFrame, list]: Depending on `return_df`, either a DataFrame with columns ["query_ticker",
      "sample_ticker", "count", "p_value", "query_ticker_idx", "sample_ticker_idx"] or a list of tuples with each
      tuple containing the query ticker, sample ticker, cooccurrence count, and p-value for the significance of
      their cooccurrence.

    The resulting DataFrame or list provides detailed insights into the pairwise relationships between stock tickers,
    including the count of their cooccurrences and the statistical significance of these counts, which can be used
    for further analysis in financial studies, such as network analysis or identifying potentially correlated assets.
    """
    tuples = []
    # -- Build the cooccurrence matrix as a polars frame: the first column
    # - "ticker" holds the row (sample) ticker labels and each remaining column
    # - is one query ticker. Missing entries are filled with 0 (mirrors the
    # - pandas DataFrame(dict-of-dicts).fillna(0) layout).
    row_tickers = []
    seen = set()
    for inner in positive_sample_distributions.values():
        for key in inner.keys():
            if key not in seen:
                seen.add(key)
                row_tickers.append(key)
    matrix_data = {"ticker": row_tickers}
    for query, inner in positive_sample_distributions.items():
        matrix_data[query] = [inner.get(rt, 0) for rt in row_tickers]
    positive_sample_distributions_df = pl.DataFrame(matrix_data)

    for ticker in tqdm(positive_sample_distributions.keys()):
        for t, c in positive_sample_distributions[ticker].items():
            p_value_positive = test_ticker_cooccurrence_significance(
                ticker,
                t,
                positive_sample_distributions_df,
                test_direction="positive_samples",
            )

            tuples.append((ticker, t, c, p_value_positive))

    if return_dataframe:
        pairwise_p_value_df = pl.DataFrame(
            tuples,
            schema=["query_ticker", "sample_ticker", "count", "p_value"],
            orient="row",
        )
        pairwise_p_value_df = pairwise_p_value_df.with_columns(
            pl.col("query_ticker")
            .replace_strict(data.ticker2idx)
            .alias("query_ticker_idx"),
            pl.col("sample_ticker")
            .replace_strict(data.ticker2idx)
            .alias("sample_ticker_idx"),
        )
        return pairwise_p_value_df
    else:
        return tuples


def get_sampling_distribution(
    ticker: str,
    pairwise_p_value_df: pl.DataFrame,
    sample_type: Literal["positive_samples", "negative_samples"] = "positive_samples",
    power: int = 1,
    positive_threshold: float = 0.1,
    negative_threshold: float = 0.1,
) -> np.ndarray:
    """
    Generates a sampling distribution for a specified ticker based on a DataFrame of sample data.

    Parameters:
    - ticker (str): The ticker symbol for which the sampling distribution is generated.
    - pairwise_p_value_df (pl.DataFrame): A DataFrame containing sample data. Expected columns include
      'query_ticker', 'sample_ticker', 'count', and 'p_value'.
    - sample_type (Literal["positive_samples", "negative_samples"], optional): Determines the type of samples to include
      in the distribution. 'positive_samples' selects samples with p_value below positive_threshold,
      'negative_samples' selects samples with 1 - p_value below negative_threshold. Defaults to 'positive_samples'.
    - power (int, optional): The power to which the probability values are raised to emphasize higher values.
      Defaults to 1, meaning no emphasis is added.
    - positive_threshold (float, optional): The threshold for p_value below which samples are considered positive.
      Defaults to 0.1.
    - negative_threshold (float, optional): The threshold for 1 - p_value below which samples are considered negative.
      Defaults to 0.1.

    Returns:
    - np.ndarray: An array of the filtered distribution containing the 'query_ticker', 'sample_ticker', and the
      calculated probability ('pos_prob') for each sample ticker.

    Raises:
    - ValueError: If an invalid sample_type is provided.

    The function filters the DataFrame based on the specified ticker and p-value (with threshold depending on positive or negative),
    calculates the probability distribution of the 'count' column, optionally emphasizes higher probabilities by raising them to the specified
    power, and normalizes the probabilities. The result is a distribution of probabilities associated with each
    sample ticker related to the query ticker.
    """
    filtered_df = pairwise_p_value_df.filter(pl.col("query_ticker") == ticker)
    if sample_type == "positive_samples":
        filtered_df = filtered_df.filter(pl.col("p_value") < positive_threshold)
    elif sample_type == "negative_samples":
        filtered_df = filtered_df.filter(
            1 - pl.col("p_value") < negative_threshold
        )
        # -- Make it so that lower counts will have high pos_prob
        max_count = filtered_df["count"].max()
        filtered_df = filtered_df.with_columns(
            (max_count - pl.col("count")).alias("count")
        )
    else:
        raise ValueError("Invalid sample_type")
    count_total = filtered_df["count"].sum()
    filtered_df = filtered_df.with_columns(
        (pl.col("count") / count_total).alias("pos_prob")
    )
    if power > 1:
        # -- Raise to a power to make high values more prominent in the resulting distribution
        prob_total = (filtered_df["pos_prob"] ** power).sum()
        filtered_df = filtered_df.with_columns(
            ((pl.col("pos_prob") ** power) / prob_total).alias("pos_prob")
        )

    filtered_distribution = filtered_df.select(
        ["query_ticker", "sample_ticker", "pos_prob"]
    ).to_numpy()

    return filtered_distribution


def sample_tgt_context_sets_from_distribution(
    filtered_distribution: np.ndarray, n_samples: int, sample_size: int
) -> list:
    """
    Samples target-context sets from a filtered distribution for a given number of samples and sample size.

    This function is designed to sample sets of 'sample_ticker' elements based on their associated probabilities
    from a pre-filtered distribution related to a single 'query_ticker'. It checks for the uniqueness of the
    'query_ticker' in the filtered distribution, verifies the sufficiency of elements for sampling, and performs
    the sampling process based on the probabilities of each 'sample_ticker'.

    Parameters:
    - filtered_distribution (np.ndarray): An array containing the filtered distribution output from
      get_sampling_distribution function, which includes 'query_ticker', 'sample_ticker', and their associated
      probabilities ('pos_prob').
    - n_samples (int): The number of sample sets to generate.
    - sample_size (int): The number of elements to include in each sample set.

    Returns:
    - list: A list of tuples, where each tuple contains the 'query_ticker' and a list of sampled 'sample_ticker'
      elements based on their probabilities.

    Raises:
    - ValueError: If the filtered distribution does not pertain to a single 'query_ticker' or if there are not enough
      elements in the distribution to fulfill the requested sample size for the 'query_ticker'.

    This function enables the generation of multiple target-context sets where each set represents a possible
    sampling of 'sample_ticker' elements associated with the single 'query_ticker'. These sets are sampled without
    replacement, ensuring diversity in the sampled elements across the generated sets.
    """
    # Unique first elements
    unique_first_elements = np.unique(filtered_distribution[:, 0])
    if len(unique_first_elements) > 1:
        raise ValueError("Should only be a single distribution")
    first_elem = unique_first_elements[0]

    sampled_pairs = []

    # Filter distribution for the current first element
    current_distribution = filtered_distribution[
        filtered_distribution[:, 0] == first_elem
    ]

    # Check if there are enough elements to sample
    if len(current_distribution) < sample_size:
        raise ValueError(
            f"Not enough elements to sample {sample_size} times for '{first_elem}'"
        )

    # Extract second elements and probabilities
    second_elements = current_distribution[:, 1]
    probabilities = current_distribution[:, 2].astype(float)

    # Sample multiple times
    # samples_for_first_elem = []
    for _ in range(n_samples):
        # Sample without replacement
        sampled_indices = np.random.choice(
            len(probabilities), size=sample_size, replace=False, p=probabilities
        )
        sampled_second_elements = second_elements[sampled_indices]
        # samples_for_first_elem.append(list(sampled_second_elements))

        # Append to result
        sampled_pairs.append((first_elem, list(sampled_second_elements)))

    return sampled_pairs
