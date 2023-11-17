from collections import defaultdict

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sktime.datasets import load_UCR_UEA_dataset


class UCR_Data:
    NUM_FROM_CLASS = 5
    PLOTLY_COLORS = px.colors.qualitative.Plotly
    PLOTLY_TEMPLATE = "plotly_dark"
    DATA_LOAD_TYPE = "numpy2d"

    def __init__(self, name: str) -> None:
        self.load_data(name)
        self.process_data()

    def load_data(self, name: str) -> None:
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

    def plot_fig(self, diff: bool = False) -> go.Figure:
        classes = self._get_classes()
        fig = go.Figure()
        for c, idxs in classes.items():
            idxs_subset = np.random.choice(idxs, self.NUM_FROM_CLASS)
            for i in idxs_subset:
                self._add_trace_to_fig(fig, c, i, diff)
        fig.update_layout(template=self.PLOTLY_TEMPLATE)
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
    def sktime_from_numpy(X: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(
            {"time_series": [pd.Series(X[i]) for i in range(X.shape[0])]}
        )

    def __str__(self) -> str:
        return f"UCR Data: {self.length} length, {self.n_classes} classes"
