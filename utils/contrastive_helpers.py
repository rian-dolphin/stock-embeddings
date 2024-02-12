from typing import Literal

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import norm
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.returns_data_class import ReturnsData


class MultiPosNegDataset(Dataset):
    def __init__(self, index_samples):
        self.index_samples = index_samples

    def __len__(self):
        return len(self.index_samples)

    def __getitem__(self, idx):
        anchor_idx, positive_indices, negative_indices = self.index_samples[idx]
        positive_indices_tensor = torch.tensor(positive_indices)
        negative_indices_tensor = torch.tensor(negative_indices)
        return anchor_idx, positive_indices_tensor, negative_indices_tensor


class BaseContrastiveLoss(nn.Module):
    def __init__(self, positive_weight=1, negative_weight=1):
        super(BaseContrastiveLoss, self).__init__()
        self.positive_weight = positive_weight
        self.negative_weight = negative_weight

    def forward(
        self,
        anchor_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
        negative_embeddings: torch.Tensor,
    ):
        """
        Args:
            anchor_embeddings (torch.Tensor): shape (batch_size, embedding_dim)
            positive_embeddings (torch.Tensor): (batch_size, num_pos_samples, embedding_dim)
            negative_embeddings (torch.Tensor): (batch_size, num_neg_samples, embedding_dim)

        Returns:
            torch.Tensor: Containing single loss value
        """
        raise NotImplementedError("This method should be overridden by subclasses.")


# class ClassificationBaseline(BaseContrastiveLoss):
#     def __init__(self, positive_weight=1, negative_weight=1):
#         super(IndividualSigmoidLoss, self).__init__()
#         self.positive_weight = positive_weight
#         self.negative_weight = negative_weight
#         self.BCELogitsCriterion = torch.nn.BCEWithLogitsLoss()

#     def forward(
#         self, anchor_embeddings, positive_embeddings, negative_embeddings
#     ) -> torch.Tensor:
#         # -- This extracts the relevant rows of the embedding matrix
#         # - Equivalent to W^T x_i in "word2vec Parameter Learning Explained"
#         # (batch_size, n_time_series, embed_dim)
#         context_embeddings = self.embeddings(inputs)

#         # -- Compute the hidden layer by a simple mean
#         hidden = context_embeddings.mean(axis=1)  # (n_time_series, embed_dim)
#         # -- Compute dot product of hidden with embeddings
#         out = torch.einsum("nd,bd->bn", self.embeddings.weight, hidden)

#         # -- Return the log softmax since we use NLLLoss loss function
#         nn.functional.log_softmax(out, dim=1)  # (batch_size, n_time_series)

#         # Only use one positive example

#         # hidden: (batch_size, embed_dim)
#         hidden = torch.mean(positive_embeddings, dim=1)

#         positive_scores = torch.einsum(
#             "bd,bd->b", [torch.mean(positive_embeddings, dim=1), anchor_embeddings]
#         )
#         negative_scores

#         positive_scores = torch.einsum(
#             "bpd,bd->bp", [positive_embeddings, anchor_embeddings]
#         )
#         negative_scores = torch.einsum(
#             "bnd,bd->bn", [negative_embeddings, anchor_embeddings]
#         )
#         # positive_loss = - torch.sum(torch.nn.functional.logsigmoid(positive_scores), dim=1)
#         # negative_loss = - torch.sum(torch.log(1-torch.sigmoid(negative_scores)), dim=1)
#         positive_loss = self.BCELogitsCriterion(
#             positive_scores, torch.ones_like(positive_scores)
#         )
#         negative_loss = self.BCELogitsCriterion(
#             negative_scores, torch.zeros_like(negative_scores)
#         )
#         loss = (
#             self.positive_weight * positive_loss + self.negative_weight * negative_loss
#         )
#         self.positive_negative_ratio = positive_loss / negative_loss

#         return loss


class IndividualSigmoidLoss(BaseContrastiveLoss):
    def __init__(self, positive_weight=1, negative_weight=1):
        super(IndividualSigmoidLoss, self).__init__()
        self.positive_weight = positive_weight
        self.negative_weight = negative_weight
        self.BCELogitsCriterion = torch.nn.BCEWithLogitsLoss()

    def forward(
        self, anchor_embeddings, positive_embeddings, negative_embeddings
    ) -> torch.Tensor:
        positive_scores = torch.einsum(
            "bpd,bd->bp", [positive_embeddings, anchor_embeddings]
        )
        negative_scores = torch.einsum(
            "bnd,bd->bn", [negative_embeddings, anchor_embeddings]
        )
        # positive_loss = - torch.sum(torch.nn.functional.logsigmoid(positive_scores), dim=1)
        # negative_loss = - torch.sum(torch.log(1-torch.sigmoid(negative_scores)), dim=1)
        positive_loss = self.positive_weight * self.BCELogitsCriterion(
            positive_scores, torch.ones_like(positive_scores)
        )
        negative_loss = self.negative_weight * self.BCELogitsCriterion(
            negative_scores, torch.zeros_like(negative_scores)
        )
        loss = positive_loss + negative_loss

        return loss, positive_loss, negative_loss


class AggregateSigmoidLoss(BaseContrastiveLoss):
    def __init__(self, positive_weight=1, negative_weight=1):
        super(AggregateSigmoidLoss, self).__init__()
        self.positive_weight = positive_weight
        self.negative_weight = negative_weight
        self.BCELogitsCriterion = torch.nn.BCEWithLogitsLoss()

    def forward(
        self, anchor_embeddings, positive_embeddings, negative_embeddings
    ) -> torch.Tensor:
        positive_scores = torch.einsum(
            "bd,bd->b", [torch.mean(positive_embeddings, dim=1), anchor_embeddings]
        )
        negative_scores = torch.einsum(
            "bd,bd->b", [torch.mean(negative_embeddings, dim=1), anchor_embeddings]
        )
        # positive_loss = - torch.sum(torch.nn.functional.logsigmoid(positive_scores))
        # negative_loss = - torch.sum(torch.log(1-torch.sigmoid(negative_scores)+0.00001))
        positive_loss = self.positive_weight * self.BCELogitsCriterion(
            positive_scores, torch.ones_like(positive_scores)
        )
        negative_loss = self.negative_weight * self.BCELogitsCriterion(
            negative_scores, torch.zeros_like(negative_scores)
        )

        loss = positive_loss + negative_loss

        return loss, positive_loss, negative_loss


class IndPos_AggSoftmax(BaseContrastiveLoss):
    """Individual Sigmoid Positive with Aggregate Softmax"""

    def __init__(self, positive_weight=1, negative_weight=1):
        super(IndPos_AggSoftmax, self).__init__()
        self.positive_weight = positive_weight
        self.negative_weight = negative_weight
        self.BCELogitsCriterion = torch.nn.BCEWithLogitsLoss()
        self.NLLCriterion = torch.nn.NLLLoss()

    def forward(
        self, anchor_embeddings, positive_embeddings, negative_embeddings
    ) -> torch.Tensor:
        ### POSITIVE LOSS - Sigmoid over each positive
        positive_scores = torch.einsum(
            "bpd,bd->bp", [positive_embeddings, anchor_embeddings]
        )
        positive_loss = self.positive_weight * self.BCELogitsCriterion(
            positive_scores, torch.ones_like(positive_scores)
        )

        ### NEGATIVE LOSS (and some positive) - Softmax with aggregate positive representation
        aggregate_positive_embeddings = torch.mean(
            positive_embeddings, dim=1
        )  # (batch_size, embedding_dim)
        aggregate_positive_scores = torch.einsum(
            "bd,bd->b", [aggregate_positive_embeddings, anchor_embeddings]
        )
        negative_scores = torch.einsum(
            "bnd,bd->bn", [negative_embeddings, anchor_embeddings]
        )
        # -- Make first column the aggregate positive scores
        # - concatenated.shape = (batch_size, 1+num_neg_samples)
        concatenated = torch.concat(
            (aggregate_positive_scores.unsqueeze(1), negative_scores), dim=1
        )
        # -- Create target for NLLLoss
        # zeros indicate the first element of concatenated is the target class, which corresponds to the positive sample
        target = torch.zeros(concatenated.shape[0], dtype=torch.long)

        negative_loss = self.negative_weight * self.NLLCriterion(
            torch.nn.functional.log_softmax(concatenated, dim=1), target
        )
        loss = positive_loss + negative_loss

        return loss, positive_loss, negative_loss


class AggPos_IndNeg(BaseContrastiveLoss):
    """Aggregate Positive with Individual Sigmoid for negatives"""

    def __init__(self, positive_weight=1, negative_weight=1):
        super(AggPos_IndNeg, self).__init__()
        self.positive_weight = positive_weight
        self.negative_weight = negative_weight
        self.BCELogitsCriterion = torch.nn.BCEWithLogitsLoss()

    def forward(
        self, anchor_embeddings, positive_embeddings, negative_embeddings
    ) -> torch.Tensor:
        ### POSITIVE LOSS - done with aggregation
        positive_scores = torch.einsum(
            "bd,bd->b", [torch.mean(positive_embeddings, dim=1), anchor_embeddings]
        )

        positive_loss = self.positive_weight * self.BCELogitsCriterion(
            positive_scores, torch.ones_like(positive_scores)
        )

        ### NEGATIVE LOSS - Sigmoid over each negative
        negative_scores = torch.einsum(
            "bnd,bd->bn", [negative_embeddings, anchor_embeddings]
        )
        negative_loss = self.negative_weight * self.BCELogitsCriterion(
            negative_scores, torch.zeros_like(negative_scores)
        )

        loss = positive_loss + negative_loss

        return loss, positive_loss, negative_loss


class JointPos_MarginalNeg(BaseContrastiveLoss):
    def __init__(self, positive_weight=1, negative_weight=1):
        super(JointPos_MarginalNeg, self).__init__()
        self.positive_weight = positive_weight
        self.negative_weight = negative_weight
        self.BCELogitsCriterion = torch.nn.BCEWithLogitsLoss()

    def forward(
        self, anchor_embeddings, positive_embeddings, negative_embeddings
    ) -> torch.Tensor:
        pos_joint_probs_pre_sigmoid = torch.einsum(
            "bpd,bd->bp", positive_embeddings, anchor_embeddings
        )
        # -- Need to do something here to prevent losses becoming huge
        # - Product of multiple probs becomes very small
        # pos_joint_probs_pre_sigmoid = torch.prod(pos_joint_probs_pre_sigmoid, dim=1)
        pos_joint_probs_pre_sigmoid = torch.mean(pos_joint_probs_pre_sigmoid, dim=1)
        # pos_joint_probs_pre_sigmoid = torch.min(
        #     pos_joint_probs_pre_sigmoid, dim=1
        # ).values
        neg_marginal_probs_pre_sigmoid = torch.einsum(
            "bnd,bd->bn", negative_embeddings, anchor_embeddings
        )

        pos_joint_probs_pre_sigmoid = pos_joint_probs_pre_sigmoid.flatten()
        neg_marginal_probs_pre_sigmoid = neg_marginal_probs_pre_sigmoid.flatten()

        positive_loss = self.positive_weight * self.BCELogitsCriterion(
            pos_joint_probs_pre_sigmoid, torch.ones_like(pos_joint_probs_pre_sigmoid)
        )
        negative_loss = self.negative_weight * self.BCELogitsCriterion(
            neg_marginal_probs_pre_sigmoid,
            torch.zeros_like(neg_marginal_probs_pre_sigmoid),
        )

        loss = positive_loss + negative_loss
        return loss, positive_loss, negative_loss


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
        sample_count = pd.Series(all_samples).value_counts()
        sample_count.index = sample_count.index.map(data.idx2ticker)
        sample_count = sample_count.to_dict()
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
    distributions: dict,
    test_direction: Literal[
        "positive_samples", "negative_samples"
    ] = "positive_samples",
    alpha: float | None = None,
    verbose: bool = False,
) -> float | bool:
    """
    Tests the significance of the cooccurrence of two stock tickers within given distributions,
    returning either the p-value or a boolean indicating significance based on a specified alpha level.

    This function calculates the observed cooccurrence count of two specified stock tickers and compares it
    against an expected count under the assumption of equal frequency distribution. A statistical test (Z-test)
    is then performed to determine the significance of the observed cooccurrence, taking into account the specified
    direction of interest (positive or negative samples).

    Parameters:
    - t1 (str): The first stock ticker symbol.
    - t2 (str): The second stock ticker symbol, cooccurrence with t1 is tested.
    - distributions (dict): A dictionary containing the cooccurrence distributions of stock tickers.
    - test_direction (Literal["positive_samples", "negative_samples"], optional): Specifies the direction of the test;
      'positive_samples' for testing higher than expected cooccurrences, 'negative_samples' for lower.
      Defaults to 'positive_samples'.
    - alpha (float | None, optional): The significance level for determining whether the observed cooccurrence is
      statistically significant. If None, the function returns the p-value. Otherwise, it returns a boolean indicating
      whether the p-value is less than alpha. Defaults to None.
    - verbose (bool, optional): If True, prints detailed test results including observed count, expected count,
      test statistic, and p-value. Defaults to False.

    Returns:
    - Union[float, bool]: Depending on the value of alpha, either returns the p-value of the significance test
      (if alpha is None) or a boolean indicating whether the cooccurrence is statistically significant at the
      specified alpha level.

    The function is useful for analyzing relationships between stock tickers in financial data, helping to identify
    pairs of tickers that cooccur more or less frequently than would be expected by chance alone.

    Example usage: test_ticker_cooccurrence_significance("JPM", "C", positive_sample_distributions, verbose=True, test_direction="positive_samples")
    """
    # Convert to DataFrame
    df = pd.DataFrame(distributions).fillna(0)

    # Calculate total counts
    total_counts = df.sum().sum()

    # Observed count for stock ticker 1 cooccurring with stock ticker 2
    observed_count = df.loc[t2, t1]

    # Expected count under equal frequency assumption
    # (1/num_TS)*int((X.shape[1]-period)/stride) * num_pos_samples
    expected_count = df[t1].sum() / len(df)

    # Perform a test
    test_statistic = (observed_count - expected_count) / np.sqrt(
        expected_count * (1 - expected_count / total_counts)
    )
    # p_value = norm.sf(abs(z_score))  # two-tailed test
    if test_direction == "positive_samples":
        p_value = norm.sf(test_statistic)  # one-tailed test
    elif test_direction == "negative_samples":
        p_value = norm.cdf(test_statistic)

    if verbose:
        print(f"Observed Count: {observed_count}")
        print(f"Expected Count: {expected_count}")
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
) -> pd.DataFrame | list:
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
    - return_dataframe (bool, optional): If True, returns the results as a pandas DataFrame. Otherwise, returns a list
      of tuples. Defaults to True.

    Returns:
    - Union[pd.DataFrame, list]: Depending on `return_df`, either a DataFrame with columns ["query_ticker",
      "sample_ticker", "count", "p_value", "query_ticker_idx", "sample_ticker_idx"] or a list of tuples with each
      tuple containing the query ticker, sample ticker, cooccurrence count, and p-value for the significance of
      their cooccurrence.

    The resulting DataFrame or list provides detailed insights into the pairwise relationships between stock tickers,
    including the count of their cooccurrences and the statistical significance of these counts, which can be used
    for further analysis in financial studies, such as network analysis or identifying potentially correlated assets.
    """
    tuples = []

    for ticker in tqdm(data.tickers):
        for t, c in positive_sample_distributions[ticker].items():
            p_value_positive = test_ticker_cooccurrence_significance(
                ticker,
                t,
                positive_sample_distributions,
                test_direction="positive_samples",
            )

            tuples.append((ticker, t, c, p_value_positive))

    if return_dataframe:
        pairwise_p_value_df = pd.DataFrame(
            tuples, columns=["query_ticker", "sample_ticker", "count", "p_value"]
        )
        pairwise_p_value_df["query_ticker_idx"] = pairwise_p_value_df[
            "query_ticker"
        ].map(data.ticker2idx)
        pairwise_p_value_df["sample_ticker_idx"] = pairwise_p_value_df[
            "sample_ticker"
        ].map(data.ticker2idx)
        return pairwise_p_value_df
    else:
        return tuples


def get_sampling_distribution(
    ticker: str,
    pairwise_p_value_df: pd.DataFrame,
    sample_type: Literal["positive_samples", "negative_samples"] = "positive_samples",
    power: int = 1,
    positive_threshold: float = 0.1,
    negative_threshold: float = 0.1,
) -> np.ndarray:
    """
    Generates a sampling distribution for a specified ticker based on a DataFrame of sample data.

    Parameters:
    - ticker (str): The ticker symbol for which the sampling distribution is generated.
    - pairwise_p_value_df (pd.DataFrame): A DataFrame containing sample data. Expected columns include
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
    filtered_df = pairwise_p_value_df[
        pairwise_p_value_df["query_ticker"] == ticker
    ].copy()
    if sample_type == "positive_samples":
        filtered_df = filtered_df[filtered_df["p_value"] < positive_threshold].copy()
    elif sample_type == "negative_samples":
        filtered_df = filtered_df[
            1 - filtered_df["p_value"] < negative_threshold
        ].copy()
    else:
        raise ValueError("Invalid sample_type")
    filtered_df["pos_prob"] = filtered_df["count"] / filtered_df["count"].sum()
    if power > 1:
        # -- Raise to a power to make high values more prominent in the resulting distribution
        filtered_df["pos_prob"] = filtered_df["pos_prob"] ** power
        filtered_df["pos_prob"] = (
            filtered_df["pos_prob"] / filtered_df["pos_prob"].sum()
        )

    filtered_distribution = filtered_df[
        ["query_ticker", "sample_ticker", "pos_prob"]
    ].values

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
