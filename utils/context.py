from typing import List, Tuple

import numpy as np
from pandas import DataFrame
from tqdm import tqdm

# ------------------------------------------
# - Initial pointwise context implementation
# ------------------------------------------


def get_pointwise_tgt_context_sets(
    df: DataFrame,
    context_size: int = 4,
    verbose: bool = False,
    iqr_noise_reduction: bool = True,
) -> List[Tuple[int, List[int]]]:
    """
    Generate index combinations for contextual similarity analysis.

    Args:
        df (DataFrame): DataFrame for similarity analysis.
        context_size (int): Number of indices to consider as context. Must be even.
        verbose (bool): Flag to enable verbose output.

    Returns:
        List[Tuple[int, List[int]]]: Each tuple contains a target index and a list of context indices.

    Raises:
        ValueError: If context_size is not an even number.
    """
    if context_size % 2 != 0:
        raise ValueError(f"Context size must be an even number, got {context_size}")

    sorted_indices = np.argsort(df.values).astype(float)
    expanded_indices = _expand_indices(sorted_indices, context_size)

    return _generate_combinations(
        expanded_indices, context_size, verbose, iqr_noise_reduction=iqr_noise_reduction
    )


def _expand_indices(indices, context_size):
    """Expand indices to ensure full context for first and last rows."""
    front_padding = indices[:, int(context_size / 2) + 1 : context_size + 1]
    end_padding = indices[:, -context_size - 1 : -int(context_size / 2) - 1]
    return np.concatenate([front_padding, indices, end_padding], axis=1)


def _generate_combinations(indices, context_size, verbose, iqr_noise_reduction=True):
    """Generate target-context index combinations."""
    combinations = []
    middle_index = int((context_size + 1) / 2)
    start, end = int(context_size / 2), indices.shape[1] - int(context_size / 2)

    if iqr_noise_reduction:
        iqr_start = (end - start) * 0.25
        iqr_end = (end - start) * 0.75

    for i in tqdm(range(start, end), disable=not verbose):
        if iqr_noise_reduction:
            if iqr_start < i < iqr_end:
                continue
        window = indices[:, i - start : i + start + 1]
        target_indices = window[:, middle_index]
        context_indices = [list(row) for row in np.delete(window, middle_index, axis=1)]
        combinations.extend(zip(target_indices, context_indices))

    return combinations


# ----------------------------------------------------------------
# - Context with option for subwindows and multiprocessing
# ----------------------------------------------------------------

from multiprocessing import Pool, cpu_count
from typing import List, Tuple, Union


def moving_avg_std(
    a: Union[List[float], np.ndarray], m: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the moving average and moving standard deviation over a given array a with window size m.

    Parameters
    ----------
    a : Union[List[float], np.ndarray]
        The array to compute the moving average and std on.
    m : int
        The window size (default is 3).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Two arrays of the moving averages and standard deviations.

    Raises
    ------
    ValueError
        If a is not a list or np.array.
        If m is not a positive integer.
    """
    if not isinstance(a, (list, np.ndarray)):
        raise ValueError("Input 'a' should be a list or numpy array.")
    if not isinstance(m, int) or m <= 0:
        raise ValueError("Window size 'm' should be a positive integer.")

    cumsum = np.cumsum(np.insert(a, 0, 0))
    cumsum2 = np.cumsum(np.insert(np.square(a), 0, 0))

    sum_m = cumsum[m:] - cumsum[:-m]
    sum_m2 = cumsum2[m:] - cumsum2[:-m]

    moving_avg = sum_m / m
    moving_std = np.sqrt(sum_m2 / m - np.square(moving_avg))

    return moving_avg, moving_std


def compute_moving_avg_std_matrix(
    ts_array: np.ndarray, m: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the moving average and moving standard deviation matrices for each time series in the given array.

    Parameters
    ----------
    ts_array : np.ndarray
        A 2D array where each row represents a time series.
    m : int
        The window size for the moving average and standard deviation (default is 3).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Two 2D arrays representing the moving averages and moving standard deviations for each time series.

    Raises
    ------
    ValueError
        If ts_array is not a 2D array or if m is not a positive integer.
    """
    if ts_array.ndim != 2:
        raise ValueError("ts_array should be a 2D array.")

    avg_std_array = np.apply_along_axis(moving_avg_std, axis=1, arr=ts_array, m=m)
    avg_matrix = avg_std_array[:, 0, :]
    std_matrix = avg_std_array[:, 1, :]

    return avg_matrix, std_matrix


def get_tgt_context_euclidean_chunk(
    i_range, ts_array, m, k, stride, z_normalize=True, verbose=True, top_k=True
):
    # -- Flip sign in argpartition to get bottom k for negative pairs
    if top_k:
        sign = 1
    else:
        sign = -1

    if z_normalize:
        mu_matrix, sigma_matrix = compute_moving_avg_std_matrix(ts_array, m=m)
    tgt_context_sets = []
    for i in tqdm(i_range, disable=not verbose):
        for t in range(0, ts_array.shape[1] - m + 1, stride):
            query = ts_array[i][t : t + m]
            subsequences = ts_array[:, t : t + m]
            if z_normalize:
                mean = mu_matrix[:, t].reshape(-1, 1)
                std = sigma_matrix[:, t].reshape(-1, 1)
                query_normalized = (query - mean[i]) / std[i]
                subsequences_normalized = (subsequences - mean) / std
                dists = np.sqrt(
                    np.sum(
                        (subsequences_normalized - query_normalized.reshape(1, -1))
                        ** 2,
                        axis=1,
                    )
                )
            else:
                dists = np.sqrt(
                    np.sum((subsequences - query.reshape(1, -1)) ** 2, axis=1)
                )
            dists[i] = sign * np.inf
            knn = np.argpartition(sign * dists, k)[:k]
            tgt_context_sets.append((i, list(knn)))
    return tgt_context_sets


def get_tgt_context_euclidean_multiprocess(
    ts_array, m, k, stride=1, z_normalize=True, verbose=True, top_k=True
):
    num_processes = cpu_count() - 1  # Get the number of CPU cores
    pool = Pool(num_processes)

    chunk_size = ts_array.shape[0] // num_processes
    ranges = [range(i * chunk_size, (i + 1) * chunk_size) for i in range(num_processes)]
    # make sure we include any left-over elements
    if ts_array.shape[0] % num_processes != 0:
        ranges[-1] = range((num_processes - 1) * chunk_size, ts_array.shape[0])

    results = pool.starmap(
        get_tgt_context_euclidean_chunk,
        [
            (i_range, ts_array, m, k, stride, z_normalize, verbose, top_k)
            for i_range in ranges
        ],
    )
    pool.close()
    pool.join()

    print("nearly returning")
    # return results

    return [item for sublist in results for item in sublist]  # flatten the list
