import numpy as np
import pandas as pd
import plotly.express as px
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


class BaseModel(nn.Module):
    """
    Base class for stock modelling embeddings.
    """

    def __init__(self, device="cpu"):
        super(BaseModel, self).__init__()
        self.embeddings = None
        self.device = torch.device(device)

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    @classmethod
    def load_model(cls, path, device="cpu"):
        # Load the state dict from the file
        state_dict = torch.load(path, map_location=torch.device(device))

        # Infer n_time_series and embedding_dim from the embeddings layer
        # Assuming the name of the embeddings layer in the state_dict is 'embeddings.weight'
        n_time_series, embedding_dim = state_dict["embeddings.weight"].shape

        # Create an instance of the cls with the inferred dimensions
        model = cls(n_time_series, embedding_dim)
        model.to(torch.device(device))

        # Load the state dict into the model
        model.load_state_dict(state_dict)

        return model

    def to_device(self, device):
        self.device = torch.device(device)
        self.to(self.device)

    def forward(self, *input):
        raise NotImplementedError("Forward method not implemented.")

    def calculate_loss(self, output, target):
        # Implement a common method for calculating loss if applicable
        raise NotImplementedError("Loss calculation method not implemented.")

    def load_embeddings_from_numpy(self, embed_array):
        self.embeddings = nn.Embedding.from_pretrained(
            torch.from_numpy(embed_array), freeze=False
        )
        print(f"Number of Time Series: {self.embeddings.weight.shape[0]}")
        print(f"Embedding Dimension: {self.embeddings.weight.shape[1]}")

    def load_embeddings_from_csv(self, fname):
        self.load_embeddings_from_numpy(np.genfromtxt(fname=fname, delimiter=","))

    def save_embeddings_to_csv(self, fname):
        if fname.split(".")[1] != "csv":
            raise ValueError("You must include .csv in your file name")
        np.savetxt(fname, self.embeddings.weight.detach().numpy(), delimiter=",")

    @staticmethod
    def _validate_tensor(x):
        if not isinstance(x, torch.Tensor):
            raise ValueError(f"Found {type(x)} rather than torch.Tensor")

    @staticmethod
    def plot_with_dimensionality_reduction(
        embeddings: np.ndarray, labels: list, method="pca"
    ):
        # check the method chosen by the user and apply accordingly
        if method.lower() == "pca":
            # apply PCA to reduce the dimensionality of the data to 2D
            reduction = PCA(n_components=2)
        elif method.lower() == "tsne":
            # apply t-SNE to reduce the dimensionality of the data to 2D
            reduction = TSNE(n_components=2)
        else:
            raise ValueError("Invalid method. Choose 'pca' or 'tsne'.")

        reduced_data = reduction.fit_transform(embeddings)

        # convert the reduced data, class labels, and entity names to a pandas DataFrame for plotting
        df = pd.DataFrame(
            {"x": reduced_data[:, 0], "y": reduced_data[:, 1], "label": labels}
        )
        df = df.sort_values(by="label")

        # plot the data using plotly, colored by the class labels
        fig = px.scatter(df, x="x", y="y", color="label", hover_name="label")
        # Update layout for smaller margins and academic style
        fig.update_layout(
            template="simple_white",  # Simple and clean layout
            margin=dict(l=20, r=20, b=20, t=20),  # Smaller margins
            font=dict(
                family="Arial", size=12, color="black"
            ),  # Academic-style font and color
            xaxis=dict(title=f"{method.upper()} Component 1", title_font=dict(size=14)),
            yaxis=dict(title=f"{method.upper()} Component 2", title_font=dict(size=14)),
            height=300,
            width=500,
            legend=dict(
                title="Class",
                # orientation="h",
                # yanchor="bottom",
                # y=1.02,
                # xanchor="right",
                # x=1,
            ),
        )
        fig.update_traces(marker={"size": 4})
        # fig.update_layout(template='plotly_dark')
        fig.show()
