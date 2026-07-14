"""Characterization tests for utils/pointwise_context.py."""

import numpy as np
import pytest
from pandas import DataFrame

from stock_embeddings.context.pointwise import get_pointwise_tgt_context_sets


def _plain(combos):
    """Convert numpy scalars/arrays in combos to plain python ints for comparison."""
    return [(int(t), [int(c) for c in ctx]) for t, ctx in combos]


def test_odd_context_size_raises():
    df = DataFrame([[1, 2, 3, 4, 5]])
    with pytest.raises(ValueError, match="even number"):
        get_pointwise_tgt_context_sets(df, context_size=3)


def test_exact_single_row_context_size_2_iqr_false():
    # Single row [10, 40, 20, 50, 30]. argsort -> column indices sorting the row:
    #   10->0, 20->2, 30->4, 40->1, 50->3  => sorted_indices = [0, 2, 4, 1, 3]
    # context_size=2: front_padding = col[2:3]=[4], end_padding = col[-3:-2]=[4]
    #   expanded = [4, 0, 2, 4, 1, 3, 4]
    # middle_index=1, start=1, end=6. iqr disabled -> process i in 1..5, window i-1:i+2:
    #   i=1 [4,0,2] -> (0,[4,2]);  i=2 [0,2,4] -> (2,[0,4]);  i=3 [2,4,1] -> (4,[2,1])
    #   i=4 [4,1,3] -> (1,[4,3]);  i=5 [1,3,4] -> (3,[1,4])
    df = DataFrame([[10, 40, 20, 50, 30]])
    combos = get_pointwise_tgt_context_sets(
        df, context_size=2, iqr_noise_reduction=False
    )
    assert _plain(combos) == [
        (0, [4, 2]),
        (2, [0, 4]),
        (4, [2, 1]),
        (1, [4, 3]),
        (3, [1, 4]),
    ]


def test_exact_single_row_context_size_2_iqr_true():
    # Same as above but iqr_noise_reduction=True: iqr_start=1.25, iqr_end=3.75
    # -> skip i where 1.25 < i < 3.75 (i.e. i=2,3). Keep i=1,4,5.
    df = DataFrame([[10, 40, 20, 50, 30]])
    combos = get_pointwise_tgt_context_sets(
        df, context_size=2, iqr_noise_reduction=True
    )
    assert _plain(combos) == [(0, [4, 2]), (1, [4, 3]), (3, [1, 4])]


def test_context_length_and_count_iqr_false():
    # Structural property derived from the algorithm: with iqr disabled the number
    # of processed targets per row equals n_cols (padding adds exactly context_size
    # columns, so end-start == n_cols). Every context list has length context_size.
    rng = np.random.default_rng(1)
    n_rows, n_cols, cs = 6, 12, 4
    df = DataFrame(rng.normal(size=(n_rows, n_cols)))
    combos = get_pointwise_tgt_context_sets(
        df, context_size=cs, iqr_noise_reduction=False
    )
    assert len(combos) == n_rows * n_cols
    assert all(len(ctx) == cs for _, ctx in combos)
    # All targets and context indices are valid column indices.
    for tgt, ctx in combos:
        assert 0 <= int(tgt) < n_cols
        assert all(0 <= int(c) < n_cols for c in ctx)


def test_iqr_true_is_subset_count_of_false():
    # iqr filtering only removes some targets, so it yields fewer combinations.
    rng = np.random.default_rng(2)
    df = DataFrame(rng.normal(size=(5, 16)))
    n_false = len(get_pointwise_tgt_context_sets(df, context_size=4, iqr_noise_reduction=False))
    n_true = len(get_pointwise_tgt_context_sets(df, context_size=4, iqr_noise_reduction=True))
    assert 0 < n_true < n_false
    # context size still respected under iqr filtering
    combos_true = get_pointwise_tgt_context_sets(df, context_size=4, iqr_noise_reduction=True)
    assert all(len(ctx) == 4 for _, ctx in combos_true)
