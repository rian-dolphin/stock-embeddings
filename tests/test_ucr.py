"""Characterization tests for stock_embeddings/ucr.py (bake-off harness).

All synthetic, no network. Tests that do NOT need sktime run unconditionally;
anything requiring the optional `ucr` extra is guarded with a skipif so the
suite still passes without it.
"""

import importlib.util
from collections import Counter

import numpy as np
import polars as pl
import pytest

from stock_embeddings.ucr import (
    add_meta_info,
    evaluate_model_sklearn,
    evaluate_resampling_UCR,
    highlight_max_in_dataset,
    initialize_eval_df,
    stratified_resample,
    to_latex,
)

_HAS_SKTIME = importlib.util.find_spec("sktime") is not None
sktime_required = pytest.mark.skipif(
    not _HAS_SKTIME, reason="requires the optional 'ucr' extra (sktime)"
)


def _blobs(seed=0, n_per=30, dim=4):
    rng = np.random.default_rng(seed)
    c0 = rng.normal(loc=-5, scale=0.5, size=(n_per, dim))
    c1 = rng.normal(loc=5, scale=0.5, size=(n_per, dim))
    X = np.vstack([c0, c1])
    y = np.array([0] * n_per + [1] * n_per)
    return X, y


def _split(X, y, n_train, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    rng.shuffle(idx)
    tr, te = idx[:n_train], idx[n_train:]
    return X[tr], X[te], y[tr], y[te]


# ------------------------------------------------------- stratified_resample
def test_stratified_resample_fold0_via_random_state():
    # The bake-off protocol: fold 0 is the ORIGINAL split; evaluate_resampling_UCR
    # achieves this by evaluating the raw split first, then resampling with
    # random_state=i for i in range(n-1). Here we check the resample itself keeps
    # the original per-split class distribution.
    X, y = _blobs(seed=1)
    Xtr, Xte, ytr, yte = _split(X, y, n_train=40, seed=2)
    Xtr_r, ytr_r, Xte_r, yte_r = stratified_resample(
        Xtr, Xte, ytr, yte, random_state=0
    )
    # Test set retains the ORIGINAL test class distribution
    assert Counter(yte_r) == Counter(yte)
    # Train set retains original train class distribution
    assert Counter(ytr_r) == Counter(ytr)


def test_stratified_resample_preserves_total_per_class():
    X, y = _blobs(seed=3)
    Xtr, Xte, ytr, yte = _split(X, y, n_train=35, seed=4)
    Xtr_r, ytr_r, Xte_r, yte_r = stratified_resample(
        Xtr, Xte, ytr, yte, random_state=7
    )
    combined = Counter(np.concatenate([ytr_r, yte_r]))
    original = Counter(np.concatenate([ytr, yte]))
    assert combined == original


def test_stratified_resample_deterministic_with_random_state():
    X, y = _blobs(seed=5)
    Xtr, Xte, ytr, yte = _split(X, y, n_train=40, seed=6)
    a = stratified_resample(Xtr, Xte, ytr, yte, random_state=11)
    b = stratified_resample(Xtr, Xte, ytr, yte, random_state=11)
    for arr_a, arr_b in zip(a, b):
        assert np.array_equal(arr_a, arr_b)


def test_stratified_resample_shapes_match_original():
    X, y = _blobs(seed=8)
    Xtr, Xte, ytr, yte = _split(X, y, n_train=40, seed=9)
    Xtr_r, ytr_r, Xte_r, yte_r = stratified_resample(
        Xtr, Xte, ytr, yte, random_state=0
    )
    assert Xtr_r.shape[0] == Xtr.shape[0]
    assert Xte_r.shape[0] == Xte.shape[0]


# ------------------------------------------------------ evaluate_model_sklearn
def test_evaluate_model_sklearn_separable_blobs_perfect():
    from sklearn.neighbors import KNeighborsClassifier

    X, y = _blobs(seed=10)
    Xtr, Xte, ytr, yte = _split(X, y, n_train=40, seed=11)
    report, acc = evaluate_model_sklearn(
        Xtr, Xte, ytr, yte,
        classifier=KNeighborsClassifier(n_neighbors=1),
        over_sampling=True,
    )
    assert acc == pytest.approx(1.0)
    assert report["accuracy"] == pytest.approx(1.0)
    assert "weighted avg" in report


def test_evaluate_model_sklearn_scale_option_runs():
    from sklearn.linear_model import LogisticRegression

    X, y = _blobs(seed=12)
    Xtr, Xte, ytr, yte = _split(X, y, n_train=40, seed=13)
    report, acc = evaluate_model_sklearn(
        Xtr, Xte, ytr, yte,
        classifier=LogisticRegression(),
        scale=True,
        over_sampling=False,
    )
    assert 0.0 <= acc <= 1.0


# ---------------------------------------------------- evaluate_resampling_UCR
def test_evaluate_resampling_UCR_structure():
    from sklearn.neighbors import KNeighborsClassifier

    X, y = _blobs(seed=14)
    Xtr, Xte, ytr, yte = _split(X, y, n_train=40, seed=15)
    agg, reports, accuracies = evaluate_resampling_UCR(
        Xtr, Xte, ytr, yte,
        n_resamples=5,
        classifier=KNeighborsClassifier(n_neighbors=1),
        verbose=False,
    )
    assert set(agg.keys()) == {"precision", "recall", "f1-score", "accuracy"}
    assert len(accuracies) == 5
    assert len(reports) == 5
    assert agg["accuracy"] == pytest.approx(np.mean(accuracies))
    # separable -> all folds perfect
    assert all(a == pytest.approx(1.0) for a in accuracies)


# ----------------------------------------------------------- result frame
def test_initialize_eval_df_is_empty_polars():
    df = initialize_eval_df()
    assert isinstance(df, pl.DataFrame)
    assert df.height == 0
    assert "dataset" in df.columns
    assert "method" in df.columns


def test_highlight_max_in_dataset_flags_per_dataset_max():
    df = pl.DataFrame(
        {
            "dataset": ["A", "A", "B", "B"],
            "method": ["m1", "m2", "m1", "m2"],
            "precision": [0.5, 0.9, 0.7, 0.3],
            "recall": [0.1, 0.2, 0.4, 0.4],
            "f1-score": [0.5, 0.9, 0.7, 0.3],
            "accuracy": [0.5, 0.9, 0.7, 0.3],
        }
    )
    out = highlight_max_in_dataset(df, cols=("precision",))
    flags = out["max_precision"].to_list()
    # A: 0.9 is max -> row1; B: 0.7 is max -> row2
    assert flags == [False, True, True, False]


def test_to_latex_produces_tabular():
    df = pl.DataFrame({"dataset": ["A"], "accuracy": [0.1234567]})
    latex = to_latex(df, float_precision=4)
    assert "\\begin{tabular}" in latex
    assert "\\end{tabular}" in latex
    assert "0.1235" in latex  # rounded to 4 dp
    assert "dataset" in latex


# ----------------------------------------------- add_meta_info graceful skip
def test_add_meta_info_missing_csv_returns_unchanged(monkeypatch, tmp_path, recwarn):
    # No Data/ dir on the resolution path -> warn + return df unchanged.
    monkeypatch.chdir(tmp_path)
    df = pl.DataFrame(
        {"dataset": ["Foo"], "method": ["m1"], "accuracy": [0.5]}
    )
    out = add_meta_info(df)  # meta_df=None -> tries to load, fails gracefully
    assert out.equals(df)
    assert any("not found" in str(w.message) for w in recwarn.list)


def test_add_meta_info_with_provided_meta():
    df = pl.DataFrame(
        {"dataset": ["Foo", "Bar"], "method": ["m1", "m1"], "accuracy": [0.5, 0.6]}
    )
    meta = pl.DataFrame(
        {
            "dataset": ["foo", "bar"],
            "Train": [10, 20],
            "Test": [5, 8],
            "Length": [100, 200],
        }
    )
    out = add_meta_info(df, meta_df=meta)
    assert "Train Size" in out.columns
    assert "Test Size" in out.columns
    assert "Length" in out.columns
    row_foo = out.filter(pl.col("dataset") == "Foo")
    assert row_foo["Train Size"][0] == 10
    assert row_foo["Length"][0] == 100


# --------------------------------------------------- sktime-gated smoke tests
@sktime_required
def test_clean_features_drops_constant_and_nan_columns():
    from stock_embeddings.ucr import UCR_Data

    X_train = np.array(
        [[1.0, 5.0, np.nan, 2.0], [2.0, 5.0, 3.0, 8.0], [3.0, 5.0, 4.0, 1.0]]
    )
    X_test = np.array([[9.0, 5.0, 1.0, 0.0]])
    Xtr_c, Xte_c = UCR_Data.clean_features(X_train, X_test)
    # col1 constant (5), col2 has NaN -> both dropped; keep cols 0 and 3
    assert Xtr_c.shape[1] == 2
    assert Xte_c.shape[1] == 2
    assert np.array_equal(Xtr_c[:, 0], X_train[:, 0])


@sktime_required
def test_sktime_from_numpy_nested_shape():
    from stock_embeddings.ucr import UCR_Data

    X = np.arange(12, dtype=float).reshape(3, 4)
    nested = UCR_Data.sktime_from_numpy(X)
    assert nested.shape[0] == 3
    assert list(nested.columns) == ["time_series"]


@sktime_required
def test_require_sktime_returns_symbols():
    from stock_embeddings.ucr import _require_sktime

    sk = _require_sktime()
    assert "Catch22" in sk
    assert "load_UCR_UEA_dataset" in sk


@sktime_required
def test_get_eval_df_prop_rocket_uses_proposed_embeddings(monkeypatch):
    # "prop+rocket" must concatenate the proposed embeddings with the rocket
    # features (the original code mistakenly used catch22 features). Stub the
    # feature extraction with distinct widths per transformer and record what
    # each method passes into the evaluation.
    import stock_embeddings.ucr as ucr_mod

    rng = np.random.default_rng(0)
    n_train, n_test, series_len = 12, 8, 24
    widths = {
        "Catch22": 5,
        "SummaryTransformer": 6,
        "RandomIntervalFeatureExtractor": 7,
        "Rocket": 8,
    }
    embed_dim = 4

    class StubData:
        name = "Stub"
        X_train = rng.normal(size=(n_train, series_len))
        X_test = rng.normal(size=(n_test, series_len))
        y = np.array(["a", "b"] * ((n_train + n_test) // 2))

        def get_sktime_features_unsupervised(self, X_train, X_test, transformer):
            w = widths[type(transformer).__name__]
            return rng.normal(size=(n_train, w)), rng.normal(size=(n_test, w))

    seen_widths = []

    def fake_eval(X_train, X_test, y_train, y_test, **kwargs):
        assert X_train.shape[1] == X_test.shape[1]
        seen_widths.append(X_train.shape[1])
        report = {
            "precision": 1.0,
            "recall": 1.0,
            "f1-score": 1.0,
            "accuracy": 1.0,
            "classification_time": 0.0,
            "feature_time": 0.0,
        }
        return report, None, None

    monkeypatch.setattr(ucr_mod, "evaluate_resampling_UCR", fake_eval)
    embeddings = rng.normal(size=(n_train + n_test, embed_dim))
    df = ucr_mod.get_eval_df(StubData(), embeddings, n_resamples=1)

    # feature_pairs call order is fixed in get_eval_df; each feature pair is
    # evaluated once per classifier ("<feature> <clf>" labels), grouped
    # consecutively. Map widths via the call order, not the (sorted) frame.
    call_order = ["proposed", "c22", "summary", "rfe", "rocket", "raw", "prop+rocket"]
    assert len(seen_widths) % len(call_order) == 0
    n_clf = len(seen_widths) // len(call_order)
    assert {m.split(" ")[0] for m in df["method"].to_list()} == set(call_order)
    groups = [seen_widths[i * n_clf : (i + 1) * n_clf] for i in range(len(call_order))]
    assert all(len(set(g)) == 1 for g in groups)  # constant width per feature
    by_method = dict(zip(call_order, (g[0] for g in groups)))
    assert by_method["proposed"] == embed_dim
    assert by_method["rocket"] == widths["Rocket"]
    assert by_method["prop+rocket"] == embed_dim + widths["Rocket"]
