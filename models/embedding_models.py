from collections import defaultdict
from typing import List, Literal, Tuple

import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.nn.functional as F
from dtaidistance import dtw
from tqdm import tqdm

from models.base_model import BaseModel


class ClassificationEmbeddings(BaseModel):
    """
    Model architecture similar to CBOW Word2Vec but adapted for stock modelling.
    Can be thought of as classifying the target stock given context.ß
    """

    def __init__(self, n_time_series: int, embedding_dim: int):
        super(ClassificationEmbeddings, self).__init__()
        self.embeddings = nn.Embedding(n_time_series, embedding_dim)

    def forward(self, inputs):
        # -- This extracts the relevant rows of the embedding matrix
        # - Equivalent to W^T x_i in "word2vec Parameter Learning Explained"
        # (batch_size, n_time_series, embed_dim)
        context_embeddings = self.embeddings(inputs)

        # -- Compute the hidden layer by a simple mean
        hidden = context_embeddings.mean(axis=1)  # (n_time_series, embed_dim)
        # -- Compute dot product of hidden with embeddings
        out = torch.einsum("nd,bd->bn", self.embeddings.weight, hidden)

        # -- Return the log softmax since we use NLLLoss loss function
        return nn.functional.log_softmax(out, dim=1)  # (batch_size, n_time_series)


class MatrixFactorization(BaseModel):
    """
    A matrix factorization model for calculating pairwise similarities.
    """

    def __init__(
        self, n_time_series: int, embedding_dim: int, normalize: bool = False
    ) -> None:
        super().__init__()
        self.n_time_series: int = n_time_series
        self.embedding_dim: int = embedding_dim
        self.embeddings = nn.Embedding(n_time_series, embedding_dim)
        if normalize:
            self.normalize_embeddings()

    def normalize_embeddings(self) -> None:
        norms = self.embeddings.weight.norm(dim=1, keepdim=True)
        self.embeddings.weight = torch.nn.Parameter(self.embeddings.weight / norms)

    def forward(self) -> torch.Tensor:
        pairwise_similarities = torch.einsum(
            "nd,md->nm", self.embeddings.weight, self.embeddings.weight
        )  # (n_time_series, n_time_series)
        return pairwise_similarities

    @staticmethod
    def apply_excluding_diagonal(array: torch.Tensor, f=np.mean):
        """Calculate the mean of a square numpy array excluding the diagonal elements."""
        MatrixFactorization._validate_tensor(array)
        if array.ndim != 2 or array.shape[0] != array.shape[1]:
            raise ValueError("Input must be a square matrix.")
        mask = torch.eye(array.shape[0], dtype=bool)
        return f(array[~mask])

    @staticmethod
    def bespoke_min_max_scale(square_matrix: torch.Tensor, exclude_diagonal=True):
        if not exclude_diagonal:
            raise NotImplementedError("Not implemented")
        off_diagonal_elements = square_matrix[np.triu_indices_from(square_matrix, k=1)]
        scaled = (square_matrix - off_diagonal_elements.min()) / (
            square_matrix.max() - square_matrix.min()
        )
        return 2 * scaled - 1

    @staticmethod
    def bespoke_normal_scale(square_matrix: torch.Tensor, exclude_diagonal=True):
        if not exclude_diagonal:
            raise NotImplementedError("Not implemented")
        return (
            square_matrix
            - MatrixFactorization.apply_excluding_diagonal(square_matrix, f=torch.mean)
        ) / MatrixFactorization.apply_excluding_diagonal(square_matrix, f=torch.std)

    @staticmethod
    def handle_scaling(
        square_matrix: torch.Tensor,
        scaled: Literal["min_max", "normal", False],
        exclude_diagonal: bool = True,
    ):
        if not exclude_diagonal:
            raise NotImplementedError("Not implemented with diagonal")
        if not scaled:
            return square_matrix
        elif scaled == "min_max":
            return MatrixFactorization.bespoke_min_max_scale(
                square_matrix, exclude_diagonal=exclude_diagonal
            )
        elif scaled == "normal":
            return MatrixFactorization.bespoke_normal_scale(
                square_matrix, exclude_diagonal=exclude_diagonal
            )
        else:
            raise ValueError("`scaled` must be one of [False, 'min_max', 'normal']")

    @staticmethod
    def get_correlation_matrix(
        X: np.ndarray, scaled: Literal["min_max", "normal", False]
    ):
        correlation_matrix = torch.tensor(np.corrcoef(X))
        if not scaled:
            return correlation_matrix
        else:
            return MatrixFactorization.handle_scaling(
                correlation_matrix, scaled=scaled, exclude_diagonal=True
            )

    @staticmethod
    def get_dtw_matrix(
        X: np.ndarray,
        max_warping_window: int | None = None,
        return_similarity: bool = False,
        scaled: Literal["min_max", "normal", False] = False,
        verbose=True,
    ):
        if verbose & (not return_similarity):
            print(
                "Warning: This returns a distance matrix not a similarity matrix. Use `return_similarity=True`for corresponding similarity matrix"
            )
        return_matrix = dtw.distance_matrix_fast(
            s=X, max_length_diff=max_warping_window
        )
        if scaled:
            return_matrix = MatrixFactorization.handle_scaling(
                return_matrix, scaled=scaled, exclude_diagonal=True
            )
        if return_similarity:
            return_matrix = -return_matrix
            np.fill_diagonal(return_matrix, np.inf)
            return return_matrix
        else:
            return return_matrix

    def calculate_loss(
        self,
        similarity_matrix1: torch.Tensor,
        similarity_matrix2: torch.Tensor,
        loss_function=F.l1_loss,
    ):
        """Calculate loss between corresponding non-diagonal elements
        of two square matrices using a specified loss function.

        This function assumes that similarity_matrix1 and similarity_matrix2 are square matrices
        of the same size.

        Args:
            similarity_matrix1 (torch.Tensor): A square matrix of shape (n, n).
            similarity_matrix2 (torch.Tensor): Another square matrix of shape (n, n).
            loss_function (callable): PyTorch loss function to be used for calculation.

        Returns:
            torch.Tensor: The calculated loss.
        """
        # Check if inputs are numpy arrays
        if isinstance(similarity_matrix1, np.ndarray) or isinstance(
            similarity_matrix2, np.ndarray
        ):
            raise TypeError(
                "similarity_matrix1 and similarity_matrix2 must be torch tensors, not numpy arrays"
            )

        # Ensure that the input matrices are square and of the same size
        assert (
            similarity_matrix1.shape == similarity_matrix2.shape
        ), "Matrices must be of the same size."
        assert (
            similarity_matrix1.shape[0] == similarity_matrix1.shape[1]
        ), "Matrices must be square."

        # Create a mask for non-diagonal elements
        n = similarity_matrix1.shape[0]
        mask = torch.eye(n, dtype=torch.bool).logical_not()

        # Apply the mask
        masked_sim_matrix1 = similarity_matrix1[mask]
        masked_sim_matrix2 = similarity_matrix2[mask]

        # Calculate the loss using the provided loss function
        loss_value = loss_function(
            masked_sim_matrix1, masked_sim_matrix2, reduction="sum"
        )

        return loss_value

    @staticmethod
    def train_MF_model(
        n_time_series: int,
        similarity_matrix: torch.Tensor,
        embedding_dim: int = 20,
        learning_rate: float = 0.02,
        epochs: int = 300,
        regularization_loss_weight: float = 0.1,
        pairwise_loss_weight: float = 0.1,
        verbose: bool = True,
    ) -> Tuple["MatrixFactorization", List[Tuple[float, float, float]], List[float]]:
        """
        Trains the MatrixFactorization model.

        Args:
            n_time_series: Number of time series in the dataset (train and test)
            similarity_matrix: The similarity matrix tensor used for training.
            embedding_dim: Dimension of the embedding space.
            learning_rate: Learning rate for the optimizer.
            epochs: Number of training epochs.
            regularization_loss_weight: Weight of the regularization loss.
            pairwise_loss_weight: Weight of the pairwise loss.
            verbose: If True, print verbose messages during training.

        Returns:
            A tuple containing the trained model, list of losses, and learning rates.
        """

        # Initialize model
        model = MatrixFactorization(
            n_time_series=n_time_series, embedding_dim=embedding_dim, normalize=True
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.8, patience=10, threshold=0.1
        )

        # Function to calculate losses
        def get_all_losses(
            model: "MatrixFactorization", correlations: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            pairwise_similarity = model()
            pairwise_loss = pairwise_loss_weight * model.calculate_loss(
                correlations,
                pairwise_similarity,
                loss_function=torch.nn.functional.l1_loss,
            )
            regularization_loss = (
                regularization_loss_weight
                * torch.abs(torch.linalg.norm(model.embeddings.weight, dim=1) - 1).sum()
            )
            total_loss = pairwise_loss + regularization_loss
            return total_loss, pairwise_loss, regularization_loss

        # Training loop
        losses: List[Tuple[float, float, float]] = []
        learning_rates: List[float] = []

        for epoch in tqdm(range(epochs), disable=not verbose):
            optimizer.zero_grad()
            total_loss, pairwise_loss, regularization_loss = get_all_losses(
                model, similarity_matrix
            )
            total_loss.backward()
            optimizer.step()
            scheduler.step(pairwise_loss)

            # Logging losses and learning rates
            losses.append(
                (total_loss.item(), pairwise_loss.item(), regularization_loss.item())
            )
            learning_rates.append(optimizer.param_groups[0]["lr"])

        return model, losses, learning_rates

    @staticmethod
    def plot_embedding_training(
        losses, learning_rates, verbose: bool = True, return_fig: bool = False
    ):
        # Unpack the losses
        total_losses, pairwise_losses, regularization_losses = zip(*losses)

        # Create a figure
        fig = go.Figure()

        # Add traces for pairwise and regularization losses
        fig.add_trace(
            go.Scatter(
                x=list(range(len(total_losses))),
                y=pairwise_losses,
                mode="lines",
                name="Total Loss",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=list(range(len(pairwise_losses))),
                y=pairwise_losses,
                mode="lines",
                name="Pairwise Loss",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=list(range(len(regularization_losses))),
                y=regularization_losses,
                mode="lines",
                name="Regularization Loss",
            )
        )

        # Create a secondary y-axis for the total loss
        fig.update_layout(
            yaxis=dict(title="Pairwise and Regularization Loss"),
            yaxis2=dict(title="Learning Rate", overlaying="y", side="right"),
        )

        # Add the total loss trace
        fig.add_trace(
            go.Scatter(
                x=list(range(len(learning_rates))),
                y=learning_rates,
                mode="lines",
                name="Learning rate",
                yaxis="y2",
            )
        )

        # Update layout
        fig.update_layout(
            title="Losses During Training", xaxis_title="Epoch", yaxis_title="Loss"
        )
        fig.update_layout(template="plotly_dark")

        # Show the figure
        if verbose:
            print(f"Final pairwise_loss: {pairwise_losses[-1]}")
            fig.show()
        if return_fig:
            return fig
