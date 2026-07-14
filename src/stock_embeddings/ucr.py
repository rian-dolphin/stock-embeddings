"""UCR time-series-classification harness (bake-off resampling evaluation).

Ported from the archived ``utils/ucr_helpers.py``. The methodology (resampling
folds, oversampling, scaling, metrics) is preserved verbatim. Changes relative
to the original:

* sktime is imported lazily via :func:`_require_sktime` so the core package can
  be imported without the optional ``ucr`` extra. Install it with::

      uv sync --extra ucr

* Our own result tables (``get_eval_df``) are assembled in polars instead of a
  pandas ``MultiIndex`` DataFrame. sktime transformer outputs are converted to
  numpy at the boundary exactly as the original did (``.values``).

* ``add_meta_info`` and the meta-CSV reads degrade gracefully (skip + warn) when
  the local-only ``Data/UCR_Summary.csv`` / ``Data/UCR_results.csv`` files are
  absent.

Access this module explicitly, e.g. ``from stock_embeddings.ucr import UCR_Data``;
it is intentionally NOT imported from ``stock_embeddings/__init__.py`` so that
importing the core package never requires the ``ucr`` extra.
"""

import time
import warnings
from collections import Counter
from typing import Tuple
from warnings import simplefilter

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from imblearn.over_sampling import RandomOverSampler
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

from stock_embeddings.data import _resolve_data_path


def _require_sktime():
    """Import the sktime symbols used by the harness.

    Raises a clear ImportError (naming the ``ucr`` extra) if sktime is not
    installed. sktime is only needed for the feature-extraction paths; keeping
    it lazy lets the rest of the harness (resampling, sklearn evaluation) run
    without the optional extra.
    """
    try:
        from sktime.datasets import load_UCR_UEA_dataset

        # `from_nested_to_2d_array` lives on a private path. On sktime 1.0.1 the
        # public `convert_to` has an incompatible signature, so we keep the
        # private import (its output is identical to the original code's).
        from sktime.datatypes._panel._convert import from_nested_to_2d_array
        from sktime.transformations.panel.catch22 import Catch22
        from sktime.transformations.panel.rocket import Rocket
        from sktime.transformations.panel.summarize import (
            RandomIntervalFeatureExtractor,
        )
        from sktime.transformations.series.summarize import SummaryTransformer
    except ImportError as e:
        raise ImportError(
            "The UCR feature extractors require the optional 'ucr' extra "
            "(sktime, dtaidistance). Install it with: uv sync --extra ucr"
        ) from e
    return {
        "load_UCR_UEA_dataset": load_UCR_UEA_dataset,
        "from_nested_to_2d_array": from_nested_to_2d_array,
        "Catch22": Catch22,
        "Rocket": Rocket,
        "RandomIntervalFeatureExtractor": RandomIntervalFeatureExtractor,
        "SummaryTransformer": SummaryTransformer,
    }


class UCR_Data:
    PLOTLY_COLORS = px.colors.qualitative.Plotly + px.colors.qualitative.D3
    PLOTLY_TEMPLATE = "plotly_dark"
    DATA_LOAD_TYPE = "numpy2d"

    def __init__(self, name: str) -> None:
        self.name = name
        self.load_data(name)
        self.process_data()
        self.summary = "\n".join(
            [
                f"Number of classes: {self.n_classes}",
                f"Number of training samples: {self.X_train.shape[0]}",
                f"Number of test samples: {self.X_test.shape[0]}",
                f"Length of time series: {self.X.shape[1]}",
            ]
        )

    def load_data(self, name: str) -> None:
        load_UCR_UEA_dataset = _require_sktime()["load_UCR_UEA_dataset"]
        try:
            self.X_train, self.y_train = load_UCR_UEA_dataset(
                name=name, return_type=self.DATA_LOAD_TYPE, split="train"
            )
            self.X_test, self.y_test = load_UCR_UEA_dataset(
                name=name, return_type=self.DATA_LOAD_TYPE, split="test"
            )
        except ValueError as e:
            if "Creating an ndarray from ragged nested sequences" in str(
                e
            ) or "not all series were of equal length" in str(e):
                raise ValueError(
                    "This TS data has variable length which we are not supporting."
                ) from e
            else:
                raise

    def process_data(self) -> None:
        self.X = np.concatenate([self.X_train, self.X_test], axis=0)
        self.y = np.concatenate([self.y_train, self.y_test], axis=0)
        self.n_classes = len(np.unique(self.y))
        self.length = self.X.shape[1]

    def plot_fig(self, diff: bool = False, num_from_class=5) -> go.Figure:
        classes = self._get_classes()
        fig = go.Figure()
        for c, idxs in classes.items():
            idxs_subset = np.random.choice(idxs, num_from_class)
            for i in idxs_subset:
                self._add_trace_to_fig(fig, c, i, diff)
        fig.update_layout(
            template=self.PLOTLY_TEMPLATE,
            height=300,
            width=600,
            xaxis=dict(title="Time"),
            yaxis=dict(title="Value"),
            margin=dict(l=20, r=20, t=20, b=20),
        )
        return fig

    def _get_classes(self) -> dict:
        classes = {}
        for c in np.unique(self.y):
            classes[c] = list(np.asarray(self.y == c).nonzero()[0])
        return classes

    def _add_trace_to_fig(self, fig: go.Figure, class_label, idx, diff: bool) -> None:
        ts = self.X[idx]
        x_axis = np.arange(len(ts) - 1) if diff else np.arange(len(ts))
        y_axis = np.diff(ts) if diff else ts
        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=y_axis,
                marker_color=self.PLOTLY_COLORS[int(class_label)],
                legendgroup=str(class_label),
                legendgrouptitle_text=f"Group {class_label}",
                name=f"{class_label}{idx}",
                opacity=0.5,
            )
        )

    @staticmethod
    def sktime_from_numpy(X: np.ndarray):
        # pandas is a transitive sktime dependency; sktime transformers consume
        # the nested-DataFrame format, so we build one at the sktime boundary.
        import pandas as pd

        return pd.DataFrame(
            {"time_series": [pd.Series(X[i]) for i in range(X.shape[0])]}
        )

    def __str__(self) -> str:
        return f"UCR Data: {self.length} length, {self.n_classes} classes"

    @staticmethod
    def get_sktime_features_unsupervised(
        X_train: np.ndarray,
        X_test: np.ndarray,
        transformer=None,
        clean=True,
        normalise=False,
    ):
        """
        Extract features from time series data using sktime transformers.

        Parameters:
        -----------
        X_train : np.ndarray
            Training data
        X_test : np.ndarray
            Test data
        transformer : sktime transformer object, default=Catch22()
            The transformer to extract features
        clean : bool, default=True
            Whether to clean features by removing constant and NaN columns
        normalize : bool, default=False
            Whether to normalize features using StandardScaler

        Returns:
        --------
        tuple
            Transformed (and optionally cleaned and normalized) training and test data
        """
        from sklearn.preprocessing import StandardScaler

        sk = _require_sktime()
        from_nested_to_2d_array = sk["from_nested_to_2d_array"]
        if transformer is None:
            transformer = sk["Catch22"]()

        transformer.fit(UCR_Data.sktime_from_numpy(X_train))
        X_train_transformed = transformer.transform(
            UCR_Data.sktime_from_numpy(X_train)
        ).values
        X_test_transformed = transformer.transform(
            UCR_Data.sktime_from_numpy(X_test)
        ).values

        # Handle if returned as sktime nested dataframe
        if X_train_transformed.shape[1] == 1:
            X_train_transformed = from_nested_to_2d_array(X_train_transformed).values
            X_test_transformed = from_nested_to_2d_array(X_test_transformed).values

        if clean:
            X_train_transformed, X_test_transformed = UCR_Data.clean_features(
                X_train_transformed, X_test_transformed
            )

        if normalise:
            scaler = StandardScaler()
            X_train_transformed = scaler.fit_transform(X_train_transformed)
            X_test_transformed = scaler.transform(X_test_transformed)

        return X_train_transformed, X_test_transformed

    @staticmethod
    def get_sktime_features_supervised(
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        transformer,
        clean=True,
        normalise=False,
    ):
        """
        Extract features from time series data using supervised sktime transformers.

        Parameters:
        -----------
        X_train : np.ndarray
            Training data
        X_test : np.ndarray
            Test data
        y_train : np.ndarray
            Training labels for supervised feature extraction
        transformer : sktime transformer object
            The supervised transformer to extract features
        clean : bool, default=True
            Whether to clean features by removing constant and NaN columns
        normalize : bool, default=False
            Whether to normalize features using StandardScaler

        Returns:
        --------
        tuple
            Transformed (and optionally cleaned and normalized) training and test data
        """
        from sklearn.preprocessing import StandardScaler

        from_nested_to_2d_array = _require_sktime()["from_nested_to_2d_array"]

        transformer.fit(UCR_Data.sktime_from_numpy(X_train), y_train)
        X_train_transformed = transformer.transform(
            UCR_Data.sktime_from_numpy(X_train)
        ).values
        X_test_transformed = transformer.transform(
            UCR_Data.sktime_from_numpy(X_test)
        ).values

        # Handle if returned as sktime nested dataframe
        if X_train_transformed.shape[1] == 1:
            X_train_transformed = from_nested_to_2d_array(X_train_transformed).values
            X_test_transformed = from_nested_to_2d_array(X_test_transformed).values

        if clean:
            X_train_transformed, X_test_transformed = UCR_Data.clean_features(
                X_train_transformed, X_test_transformed
            )

        if normalise:
            scaler = StandardScaler()
            X_train_transformed = scaler.fit_transform(X_train_transformed)
            X_test_transformed = scaler.transform(X_test_transformed)

        return X_train_transformed, X_test_transformed

    @staticmethod
    def clean_features(
        X_train: np.ndarray, X_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove columns from both training and test datasets where columns in the training set
        contain NaN values or have uniform values across all rows.

        Parameters:
        - X_train (np.ndarray): A 2D NumPy array representing the training dataset.
        - X_test (np.ndarray): A 2D NumPy array representing the test dataset.

        Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the cleaned training and test datasets.
        """
        # Identify columns with NaN values
        cols_with_nan = np.any(np.isnan(X_train), axis=0)
        # Identify columns with uniform values
        cols_with_no_variance = np.isnan(np.var(X_train, axis=0)) | (
            np.var(X_train, axis=0) == 0
        )
        # Combine columns to drop
        cols_to_drop = cols_with_nan | cols_with_no_variance
        # Drop columns from both arrays
        X_train_cleaned = X_train[:, ~cols_to_drop]
        X_test_cleaned = X_test[:, ~cols_to_drop]

        return X_train_cleaned, X_test_cleaned


def evaluate_model_sklearn(
    embeddings_train: np.ndarray,
    embeddings_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    classifier=MLPClassifier(),
    scale: bool = False,
    over_sampling: bool = True,
    verbose: bool = False,
) -> float:
    """
    Evaluate a machine learning model using scikit-learn.

    Args:
        embeddings_train: Training data embeddings.
        embeddings_test: Test data embeddings.
        y_train: Training data labels.
        y_test: Test data labels.
        classifier: The classifier to use. Defaults to MLPClassifier.
        over_sampling: Whether to apply oversampling. Defaults to True.
        verbose: Whether to print detailed classification report. Defaults to False.

    Returns:
        The accuracy of the model on the test data.
    """
    if scale:
        scaler = StandardScaler()
        scaler.fit(embeddings_train)
        embeddings_train = scaler.transform(embeddings_train)
        embeddings_test = scaler.transform(embeddings_test)
    if over_sampling:
        sm = RandomOverSampler()
        X_train_oversampled, y_train_oversampled = sm.fit_resample(
            embeddings_train, y_train
        )
        classifier.fit(X_train_oversampled, y_train_oversampled)
    else:
        classifier.fit(embeddings_train, y_train)

    # Predict and evaluate
    y_preds = classifier.predict(embeddings_test)
    report = classification_report(
        y_true=y_test, y_pred=y_preds, output_dict=True, digits=4
    )
    if verbose:
        print(classification_report(y_true=y_test, y_pred=y_preds, digits=3))
    accuracy = report["accuracy"]
    return report, accuracy


def get_kNN_accuracy_MF_UCR(data, model, k=1):
    import torch

    case_base = model.embeddings.weight.detach()[: data.X_train.shape[0], :]
    test_embeddings = model.embeddings.weight.detach()[data.X_train.shape[0] :, :]
    temp = torch.einsum("nd, md->nm", case_base, test_embeddings)
    temp = [data.y_train[i] for i in torch.argmax(temp, dim=0)]
    accuracy = np.mean(temp == data.y_test)
    return accuracy


def stratified_resample(X_train, X_test, y_train, y_test, random_state=None):
    """
    Resampling approach to follow the time series bake off paper
    https://link.springer.com/article/10.1007/s10618-016-0483-9

    "We run the same 100 resample folds on each problem for every classifier. The
    first fold is always the original train test split. The other resamples are stratified to
    retain class distribution in the original train/trest splits."
    """
    # Calculate class distribution in original train/test split
    train_dist = Counter(y_train)
    test_dist = Counter(y_test)

    X = np.concatenate([X_train, X_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)

    X_train_res, y_train_res = [], []
    X_test_res, y_test_res = [], []

    for class_value in np.unique(y):
        test_num = test_dist[class_value]
        X_class = X[y == class_value]
        y_class = y[y == class_value]
        (
            X_train_res_class,
            X_test_res_class,
            y_train_res_class,
            y_test_res_class,
        ) = train_test_split(
            X_class,
            y_class,
            test_size=test_num,
            shuffle=True,
            random_state=random_state,
        )

        X_train_res.append(X_train_res_class)
        y_train_res.append(y_train_res_class)
        X_test_res.append(X_test_res_class)
        y_test_res.append(y_test_res_class)

    return (
        np.concatenate(X_train_res),
        np.concatenate(y_train_res),
        np.concatenate(X_test_res),
        np.concatenate(y_test_res),
    )


def evaluate_resampling_UCR(
    X_train,
    X_test,
    y_train,
    y_test,
    n_resamples=50,
    classifier=KNeighborsClassifier(n_neighbors=1, metric="euclidean"),
    scale=False,
    over_sampling=True,
    verbose=True,
):
    # Store results
    accuracies = []
    reports = []

    # First fold - original train/test split
    report, accuracy = evaluate_model_sklearn(
        X_train, X_test, y_train, y_test, classifier, scale, over_sampling
    )
    accuracies.append(accuracy)
    reports.append(report["weighted avg"])

    for i in tqdm(range(n_resamples - 1), disable=not verbose):
        X_train_res, y_train_res, X_test_res, y_test_res = stratified_resample(
            X_train, X_test, y_train, y_test, random_state=i
        )

        report, accuracy = evaluate_model_sklearn(
            X_train_res,
            X_test_res,
            y_train_res,
            y_test_res,
            classifier,
            scale,
            over_sampling,
        )

        # Store results
        accuracies.append(accuracy)
        reports.append(report["weighted avg"])
    aggregate_report = {
        metric: sum(r[metric] for r in reports) / len(reports)
        for metric in ["precision", "recall", "f1-score"]
    }
    aggregate_report["accuracy"] = np.mean(accuracies)
    return aggregate_report, reports, accuracies


# Columns of the polars result frame produced by ``get_eval_df``. The original
# used a pandas MultiIndex of (dataset, method); here those are plain columns.
EVAL_DF_COLUMNS = [
    "dataset",
    "method",
    "precision",
    "recall",
    "f1-score",
    "accuracy",
    "classification_time",
    "feature_time",
]


def initialize_eval_df():
    """Return an empty polars result frame (replaces the pandas MultiIndex df)."""
    import polars as pl

    return pl.DataFrame(
        schema={
            "dataset": pl.Utf8,
            "method": pl.Utf8,
            "precision": pl.Float64,
            "recall": pl.Float64,
            "f1-score": pl.Float64,
            "accuracy": pl.Float64,
            "classification_time": pl.Float64,
            "feature_time": pl.Float64,
        }
    )


def get_eval_df(
    data: UCR_Data,
    embeddings: np.ndarray,
    df=None,
    n_resamples=20,
    verbose=False,
    scale=False,
    over_sampling=True,
    suffix="",
    suppress_warnings=True,
):
    """
    Run a test over multiple samples and compare against various baselines.

    Returns a polars DataFrame with one row per (dataset, method), sorted by
    ``dataset`` then ``method`` (replaces the original pandas ``sort_index``).
    """
    import polars as pl

    sk = _require_sktime()
    Catch22 = sk["Catch22"]
    Rocket = sk["Rocket"]
    RandomIntervalFeatureExtractor = sk["RandomIntervalFeatureExtractor"]
    SummaryTransformer = sk["SummaryTransformer"]

    if suppress_warnings:
        simplefilter("ignore", category=ConvergenceWarning)
        simplefilter("ignore", category=UndefinedMetricWarning)
    if df is None:
        df = initialize_eval_df()
    name = data.name

    train_size = data.X_train.shape[0]
    embeddings_train = embeddings[:train_size, :]
    embeddings_test = embeddings[train_size:, :]
    y_train = data.y[:train_size]
    y_test = data.y[train_size:]

    fit_times = {}
    start_time = time.time()
    c22_train, c22_test = data.get_sktime_features_unsupervised(
        X_train=data.X_train, X_test=data.X_test, transformer=Catch22()
    )
    fit_times["c22"] = time.time() - start_time
    start_time = time.time()
    summary_train, summary_test = data.get_sktime_features_unsupervised(
        X_train=data.X_train, X_test=data.X_test, transformer=SummaryTransformer()
    )
    fit_times["summary"] = time.time() - start_time
    start_time = time.time()
    rfe_train, rfe_test = data.get_sktime_features_unsupervised(
        X_train=data.X_train,
        X_test=data.X_test,
        transformer=RandomIntervalFeatureExtractor(),
    )
    fit_times["rfe"] = time.time() - start_time
    start_time = time.time()
    rocket_train, rocket_test = data.get_sktime_features_unsupervised(
        X_train=data.X_train, X_test=data.X_test, transformer=Rocket()
    )
    fit_times["rocket"] = time.time() - start_time

    def add_row_from_aggregate_report(df, name, method, aggregate_report):
        row = {"dataset": name, "method": method}
        row.update(aggregate_report)
        new_row = pl.DataFrame([row]).select(df.columns)
        return pl.concat([df, new_row], how="vertical")

    feature_pairs = (
        ("proposed", embeddings_train, embeddings_test),
        ("c22", c22_train, c22_test),
        ("summary", summary_train, summary_test),
        ("rfe", rfe_train, rfe_test),
        ("rocket", rocket_train, rocket_test),
        ("raw", data.X_train, data.X_test),
        # (
        #     "prop+c22",
        #     np.concatenate([c22_train, embeddings_train], axis=1),
        #     np.concatenate([c22_test, embeddings_test], axis=1),
        # ),
        (
            "prop+rocket",
            np.concatenate([embeddings_train, rocket_train], axis=1),
            np.concatenate([embeddings_test, rocket_test], axis=1),
        ),
    )

    classifiers = (
        # {
        #     "clf_name": "MLP",
        #     "classifier": MLPClassifier(alpha=0.001, max_iter=25),
        #     "scale": scale,
        #     "over_sampling": over_sampling,
        # },
        {
            "clf_name": "SVC",
            "classifier": SVC(kernel="rbf"),
            "scale": scale,
            "over_sampling": over_sampling,
        },
        {
            "clf_name": "logistic",
            "classifier": LogisticRegression(),
            "scale": scale,
            "over_sampling": over_sampling,
        },
        # {
        #     "clf_name": "DT",
        #     "classifier": DecisionTreeClassifier(),
        #     "scale": scale,
        #     "over_sampling": over_sampling,
        # },
        {
            "clf_name": "1NN",
            "classifier": KNeighborsClassifier(n_neighbors=1, metric="euclidean"),
            "scale": scale,
            "over_sampling": over_sampling,
        },
    )
    for feature_method, X_train, X_test in feature_pairs:
        for clf_dict in classifiers:
            clf_name = clf_dict["clf_name"]
            if verbose:
                print(f"Running {clf_name} with {feature_method}")
            # -- Add MLP
            if verbose:
                print("Computing MLP")
            start_time = time.time()
            report, _, _ = evaluate_resampling_UCR(
                X_train,
                X_test,
                y_train,
                y_test,
                n_resamples=n_resamples,
                verbose=verbose,
                **{k: v for k, v in clf_dict.items() if k != "clf_name"},
            )
            end_time = time.time()
            report["classification_time"] = end_time - start_time
            if feature_method in fit_times:
                report["feature_time"] = fit_times[feature_method]
            else:
                report["feature_time"] = np.nan
            df = add_row_from_aggregate_report(
                df, name, feature_method + f" {clf_name}" + suffix, report
            )

    return df.sort(["dataset", "method"])


def _load_meta_df(path="Data/UCR_Summary.csv"):
    """Load the UCR meta-info CSV as a polars frame, or ``None`` if absent.

    The meta CSVs (``Data/UCR_Summary.csv`` / ``Data/UCR_results.csv``) were
    local-only artifacts that this repo never shipped, so their absence is
    expected. We search upward from the cwd via the same helper used elsewhere
    in the package and warn (rather than raise) when nothing is found.
    """
    import polars as pl

    try:
        resolved = _resolve_data_path(path)
    except FileNotFoundError:
        warnings.warn(
            f"Meta-info file {path!r} not found (local-only artifact); "
            "skipping meta-info enrichment.",
            stacklevel=2,
        )
        return None
    return pl.read_csv(resolved)


def add_meta_info(df, meta_df=None):
    """Attach Train/Test/Length columns to the polars result ``df``.

    If ``meta_df`` is None we attempt to load it from ``Data/UCR_Summary.csv``.
    When the meta file is missing the original ``df`` is returned unchanged
    (with a warning) so downstream code degrades gracefully.

    ``meta_df`` is expected to have a lowercased dataset key column named
    ``dataset`` (or an unnamed first column) plus ``Train``, ``Test``,
    ``Length`` columns, matching the old pandas layout where the index was the
    lowercased dataset name.
    """
    import polars as pl

    if meta_df is None:
        meta_df = _load_meta_df()
    if meta_df is None:
        return df

    # Normalise the meta frame to columns: dataset (lowercase), Train, Test,
    # Length. The old pandas version indexed by lowercased dataset name.
    meta_cols = meta_df.columns
    key_col = "dataset" if "dataset" in meta_cols else meta_cols[0]
    meta_norm = meta_df.rename({key_col: "dataset"}).with_columns(
        pl.col("dataset").str.to_lowercase().alias("dataset")
    ).select(["dataset", "Train", "Test", "Length"])

    out = df.with_columns(pl.col("dataset").str.to_lowercase().alias("_key"))
    out = out.join(
        meta_norm.rename(
            {"Train": "Train Size", "Test": "Test Size", "Length": "Length"}
        ),
        left_on="_key",
        right_on="dataset",
        how="left",
    ).drop("_key")
    return out


def highlight_max_in_dataset(df, cols=("precision", "recall", "f1-score", "accuracy")):
    """Return, per dataset, which (row, column) cells hold the max value.

    Replaces the old pandas Styler ``background-color`` highlighting (presentation
    only) with a plain polars boolean-flag frame: one boolean column ``max_<col>``
    per metric, True where the value equals the per-dataset maximum. Computed
    numbers are unchanged.
    """
    import polars as pl

    exprs = []
    for col in cols:
        exprs.append(
            (pl.col(col) == pl.col(col).max().over("dataset")).alias(f"max_{col}")
        )
    return df.with_columns(exprs)


def to_latex(df, float_precision=4):
    """Render a polars result frame as a simple LaTeX ``tabular`` string.

    Replaces the pandas ``Styler.to_latex`` export (presentation only). No
    highlighting; just a plain booktabs-free tabular.
    """
    cols = df.columns
    n = len(cols)
    lines = [
        "\\begin{tabular}{" + "l" * n + "}",
        "\\hline",
        " & ".join(str(c) for c in cols) + " \\\\",
        "\\hline",
    ]
    for row in df.iter_rows():
        cells = []
        for v in row:
            if isinstance(v, float):
                cells.append(f"{v:.{float_precision}f}")
            else:
                cells.append("" if v is None else str(v))
        lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    return "\n".join(lines)
