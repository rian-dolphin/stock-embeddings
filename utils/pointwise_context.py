import numpy as np
from tqdm import tqdm
from typing import List, Tuple
from pandas import DataFrame


def get_contextual_indices(
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
