import numpy as np
from tqdm import tqdm
from abc import ABC, abstractmethod
from scipy.spatial.distance import pdist, squareform


class SimilarityMetric(ABC):
    """Abstract Base Class for similarity metrics."""

    def __init__(self, vectorized: bool, similarity: bool):
        self.vectorized = vectorized
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
    period=5,
    stride=None,
    context_size=4,
    verbose=True,
):
    """
    Obtain target-context sets for a given similarity metric from a dataset of time series.

    Args:
        X (np.array): Array of time series (num_time_series, num_time_periods).
        metric_class (SimilarityMetric): Metric class for computing similarities.
        period (int): Window size. Defaults to 5.
        stride (int): Step size. Defaults to period for non-overlapping windows.
        context_size (int): Number of context time series in each set. Defaults to 4.

    Returns:
        list: Target-context sets in the format [(target_index, [context_indices]), ...].
    """
    T, n_time_series = X.shape[1], X.shape[0]
    stride = stride or period
    tgt_context_sets = []

    for t in tqdm(range(0, T - period, stride), disable=not verbose):
        X_slice = X[:, t : t + period]
        sim_matrix = compute_similarity_matrix(X_slice, metric_class)
        tgt_context_sets.extend(extract_indices(sim_matrix, context_size))

    return tgt_context_sets


def compute_similarity_matrix(X_slice, metric_class):
    """
    Compute the similarity matrix for a given slice of data and a metric.

    Args:
        X_slice (np.array): Sliced data for a specific window.
        metric_class (SimilarityMetric): Metric class for computing similarities.

    Returns:
        np.array: Computed similarity matrix.
    """
    return metric_class.compute_matrix_pairwise(
        X_slice.T if isinstance(metric_class, ICCBRMetric) else X_slice
    )


def extract_indices(sim_matrix, context_size):
    """
    Extract indices of context time series based on the similarity matrix.

    Args:
        sim_matrix (np.array): Similarity matrix.
        context_size (int): Number of context time series in each set.

    Returns:
        list: List of tuples (target_index, [context_indices]).
    """
    np.fill_diagonal(sim_matrix, -1)
    indices = np.argpartition(sim_matrix, -context_size, axis=1)[:, -context_size:]
    return [(i, list(xi)) for i, xi in enumerate(indices)]
