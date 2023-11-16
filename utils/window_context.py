from abc import ABC, abstractmethod

import numpy as np
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm


class SimilarityMetric(ABC):
    """Abstract Base Class for similarity metrics."""

    def __init__(self, vectorized: bool, similarity: bool):
        self.vectorized = vectorized
        # -- True if a similarity measure, False if distance metric
        self.similarity = similarity

    @abstractmethod
    def compute(self, vector_1, vector_2):
        pass

    @abstractmethod
    def compute_matrix_pairwise(self, matrix):
        pass


class Pearson(SimilarityMetric):
    """Pearson correlation coefficient as a similarity metric."""

    def __init__(self):
        super().__init__(vectorized=False, similarity=True)

    def compute(self, vector_1, vector_2):
        return np.corrcoef(x=vector_1, y=vector_2)[0, 1]

    def compute_matrix_pairwise(self, matrix):
        return np.corrcoef(matrix)


class ICCBRMetric(SimilarityMetric):
    """ICCBR Metric."""

    def __init__(self, w=0.5):
        super().__init__(vectorized=False, similarity=True)
        self.w = w

    def compute(self, vector_1, vector_2):
        cumprod_euclidean = abs(
            np.cumprod(1 + vector_1)[-1] - np.cumprod(1 + vector_2)[-1]
        )
        nmc = self._no_mean_correlation(vector_1, vector_2)[0]
        return self.w / (1 + cumprod_euclidean) + (1 - self.w) * nmc

    def compute_matrix_pairwise(self, matrix):
        matrix = matrix.T
        m, n = matrix.shape

        similarity_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i, n):
                if i == j:
                    similarity_matrix[
                        i, j
                    ] = 1.0  # similarity of a vector to itself is 1.0
                else:
                    vector_1 = matrix[:, i]
                    vector_2 = matrix[:, j]
                    similarity = self.compute(vector_1, vector_2)
                    similarity_matrix[i, j] = similarity
                    similarity_matrix[j, i] = similarity

        return similarity_matrix

    def _no_mean_correlation(self, A, B):
        am = A  # - np.mean(A, axis=0, keepdims=True)
        bm = B  # - np.mean(B, axis=0, keepdims=True)
        return (
            am.T
            @ bm
            / (
                np.sqrt(np.sum(am**2, axis=0, keepdims=True)).T
                * np.sqrt(np.sum(bm**2, axis=0, keepdims=True))
            )
        )


class Euclidean(SimilarityMetric):
    """Euclidean distance as a metric."""

    def __init__(self, vectorized=True):
        super().__init__(vectorized, similarity=False)

    def compute(self, query_row, other_rows):
        if self.vectorized:
            query_row_repeated = np.tile(query_row, other_rows.shape[0]).reshape(
                other_rows.shape[0], -1
            )
            return np.linalg.norm(query_row_repeated - other_rows, axis=1)
        else:
            # vector_1 = query_row
            # vector_2 = other_rows
            return np.linalg.norm(query_row - other_rows)

    def compute_matrix_pairwise(self, matrix):
        # Compute the pairwise distances
        distances = pdist(matrix, "euclidean")

        # Convert the pairwise distances to a square form
        return squareform(distances)


def get_target_context_sets(
    X,
    metric_class: SimilarityMetric,
    window_length=5,
    stride=None,
    context_size=4,
    verbose=True,
):
    """
    Obtain target-context sets for a given similarity metric from a dataset of time series.

    Args:
        X (np.array): Array of time series (num_time_series, num_time_periods).
        metric_class (SimilarityMetric): Metric class for computing similarities.
        window_length (int): Window size. Defaults to 5.
        stride (int): Step size. Defaults to window_length for non-overlapping windows.
        context_size (int): Number of context time series in each set. Defaults to 4.

    Returns:
        list: Target-context sets in the format [(target_index, [context_indices]), ...].
    """
    T = X.shape[1]
    if stride is None:
        stride = window_length
    tgt_context_sets = []

    for t in tqdm(range(0, T - window_length, stride), disable=not verbose):
        X_slice = X[:, t : t + window_length]
        metric_matrix = metric_class.compute_matrix_pairwise(X_slice)
        tgt_context_sets.extend(
            extract_indices(metric_matrix, context_size, metric_class)
        )

    return tgt_context_sets


def extract_indices(
    metric_matrix: np.array, context_size: int, metric_class: SimilarityMetric
):
    """
    Extract indices of context time series based on the similarity matrix.

    Args:
        metric_matrix (np.array): Output from metric class pairwise matrix.
        context_size (int): Number of context time series in each set.
        metric_class (SimilarityMetric): Metric class - needed here to
            determine whether it is a similarity or distance for correct ordering.
    Returns:
        list: List of tuples (target_index, [context_indices]).
    """
    multiplier = 1 if metric_class.similarity else -1
    sim_matrix = multiplier * metric_matrix
    np.fill_diagonal(sim_matrix, -1)
    indices = np.argpartition(sim_matrix, -context_size, axis=1)[:, -context_size:]
    return [(i, list(xi)) for i, xi in enumerate(indices)]
    return [(i, list(xi)) for i, xi in enumerate(indices)]
