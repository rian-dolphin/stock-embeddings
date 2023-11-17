from collections import defaultdict

import numpy as np
import pandas as pd
import plotly.express as px
from sktime.datasets import load_UCR_UEA_dataset


class UCR_Data:
    def __init__(self, name) -> None:
        try:
            self.X_train, self.y_train = load_UCR_UEA_dataset(
                name=name, return_type="numpy2d", split="train"
            )
            self.X_test, self.y_test = load_UCR_UEA_dataset(
                name=name, return_type="numpy2d", split="test"
            )
        except ValueError as e:
            print(name)
            if "Creating an ndarray from ragged nested sequences" in str(e):
                raise ValueError(
                    "This TS data has variable length which we are not supporting."
                )
            elif "not all series were of equal length" in str(e):
                raise ValueError(
                    "This TS data has variable length which we are not supporting."
                )
            else:
                raise e

        self.X = np.concatenate([self.X_train, self.X_test], axis=0)
        self.y = np.concatenate([self.y_train, self.y_test], axis=0)

        self.n_classes = len(np.unique(self.y))
        self.length = self.X.shape[1]
        # self.class_dict = self.get_class_dict()

    def plot_fig(self, diff=False):
        X = self.X
        y = self.y
        # -- Get the y index for each unique class
        classes = {}
        for c in np.unique(y):
            classes[c] = list(np.asarray(y == c).nonzero()[0])

        num_from_class = 5

        import plotly.graph_objects as go

        fig = go.Figure()
        colors = px.colors.qualitative.Plotly
        for c, idxs in classes.items():
            idxs_subset = np.random.choice(idxs, num_from_class)
            for i in idxs_subset:
                ts = X[i]
                if diff:
                    fig.add_trace(
                        go.Scatter(
                            x=np.arange(len(ts) - 1),
                            y=np.diff(ts),
                            marker_color=colors[int(c)],
                            legendgroup=c,
                            legendgrouptitle_text=f"Group {c}",
                            name=c + str(i),
                            opacity=0.5,
                        )
                    )
                else:
                    fig.add_trace(
                        go.Scatter(
                            x=np.arange(len(ts)),
                            y=ts,
                            marker_color=colors[int(c)],
                            legendgroup=c,
                            legendgrouptitle_text=f"Group {c}",
                            name=c + str(i),
                            opacity=0.5,
                        )
                    )

        fig.update_layout(template="plotly_dark")

        return fig

    @staticmethod
    def sktime_from_numpy(X):
        return pd.DataFrame(
            {"time_series": [pd.Series(X[i]) for i in range(X.shape[0])]}
        )
