from collections import defaultdict
from typing import List, Literal, Tuple

import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.nn.functional as F
from dtaidistance import dtw
from sklearn.metrics.pairwise import euclidean_distances
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

    def forward(self) -> torch.Tensor:
        pairwise_similarities = torch.einsum(
            "nd,md->nm", self.embeddings.weight, self.embeddings.weight
        )  # (n_time_series, n_time_series)
        return pairwise_similarities

    @classmethod
    def apply_excluding_diagonal(cls, array: torch.Tensor, f=np.mean):
        """Calculate the mean of a square numpy array excluding the diagonal elements."""
        cls._validate_tensor(array)
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

    @classmethod
    def bespoke_normal_scale(cls, square_matrix: torch.Tensor, exclude_diagonal=True):
        if not exclude_diagonal:
            raise NotImplementedError("Not implemented")
        return (
            square_matrix - cls.apply_excluding_diagonal(square_matrix, f=torch.mean)
        ) / cls.apply_excluding_diagonal(square_matrix, f=torch.std)

    @classmethod
    def handle_scaling(
        cls,
        square_matrix: torch.Tensor,
        scaled: Literal["min_max", "normal", False],
        exclude_diagonal: bool = True,
        truncate: bool = False,
    ):
        if not scaled:
            return square_matrix
        return_matrix = square_matrix
        if not exclude_diagonal:
            raise NotImplementedError("Not implemented with diagonal")
        if not scaled:
            pass
        elif scaled == "min_max":
            return_matrix = cls.bespoke_min_max_scale(
                square_matrix, exclude_diagonal=exclude_diagonal
            )
        elif scaled == "normal":
            return_matrix = cls.bespoke_normal_scale(
                square_matrix, exclude_diagonal=exclude_diagonal
            )
        else:
            raise ValueError("`scaled` must be one of [False, 'min_max', 'normal']")
        if truncate:
            return_matrix = torch.clamp(return_matrix, min=-2, max=2)
        return return_matrix

    @classmethod
    def get_correlation_matrix(
        cls,
        X: np.ndarray,
        scaled: Literal["min_max", "normal", False],
        truncate: bool = False,
    ):
        correlation_matrix = torch.tensor(np.corrcoef(X))
        correlation_matrix = cls.handle_scaling(
            correlation_matrix, scaled=scaled, exclude_diagonal=True, truncate=truncate
        )
        correlation_matrix.fill_diagonal_(np.nan)
        return correlation_matrix

    @classmethod
    def get_euclidean_matrix(
        cls,
        X: np.ndarray,
        scaled: Literal["min_max", "normal", False],
        return_similarity: bool = False,
        verbose: bool = True,
        truncate: bool = False,
    ):
        if verbose & (not return_similarity):
            print(
                "Warning: This returns a distance matrix not a similarity matrix. Use `return_similarity=True`for corresponding similarity matrix"
            )
        return_matrix = euclidean_distances(X)
        if not isinstance(return_matrix, torch.Tensor):
            return_matrix = torch.tensor(return_matrix)
        if not scaled:
            pass
        else:
            return_matrix = cls.handle_scaling(
                return_matrix, scaled=scaled, exclude_diagonal=True, truncate=truncate
            )
        if return_similarity:
            return_matrix = -return_matrix
        return_matrix.fill_diagonal_(np.nan)
        return return_matrix

    @classmethod
    def get_dtw_matrix(
        cls,
        X: np.ndarray,
        max_warping_window: int | None = None,
        return_similarity: bool = False,
        scaled: Literal["min_max", "normal", False] = False,
        verbose=True,
        truncate: bool = False,
    ):
        if verbose & (not return_similarity):
            print(
                "Warning: This returns a distance matrix not a similarity matrix. Use `return_similarity=True`for corresponding similarity matrix"
            )
        return_matrix = dtw.distance_matrix_fast(
            s=X, max_length_diff=max_warping_window
        )
        if scaled:
            return_matrix = cls.handle_scaling(
                return_matrix, scaled=scaled, exclude_diagonal=True, truncate=truncate
            )
        if return_similarity:
            return_matrix = -return_matrix
            np.fill_diagonal(return_matrix, np.inf)
        return_matrix.fill_diagonal_(np.nan)
        return torch.tensor(return_matrix)

    def calculate_loss(
        self,
        model_similarity_matrix: torch.Tensor,
        target_similarity_matrix: torch.Tensor,
        loss_function=F.l1_loss,
        noise_mask: bool = False,
    ):
        """Calculate loss between corresponding non-diagonal elements
        of two square matrices using a specified loss function.

        This function assumes that model_similarity_matrix and target_similarity_matrix are square matrices
        of the same size.

        Args:
            model_similarity_matrix (torch.Tensor): A square matrix of shape (n, n).
            target_similarity_matrix (torch.Tensor): Another square matrix of shape (n, n).
            loss_function (callable): PyTorch loss function to be used for calculation.
            noise_mask (bool): If True then only include extreme similarity values

        Returns:
            torch.Tensor: The calculated loss.
        """
        # Check if inputs are numpy arrays
        if isinstance(model_similarity_matrix, np.ndarray) or isinstance(
            target_similarity_matrix, np.ndarray
        ):
            raise TypeError(
                "model_similarity_matrix and target_similarity_matrix must be torch tensors, not numpy arrays"
            )

        # Ensure that the input matrices are square and of the same size
        assert (
            model_similarity_matrix.shape == target_similarity_matrix.shape
        ), "Matrices must be of the same size."
        assert (
            model_similarity_matrix.shape[0] == model_similarity_matrix.shape[1]
        ), "Matrices must be square."

        # Mask diagonals which should be provided as nan
        mask = ~model_similarity_matrix.isnan() & ~target_similarity_matrix.isnan()
        # Create a mask for non-diagonal elements
        # n = model_similarity_matrix.shape[0]
        # mask = torch.eye(n, dtype=torch.bool).logical_not()

        # Apply the mask
        masked_sim_matrix1 = model_similarity_matrix[mask]
        masked_sim_matrix2 = target_similarity_matrix[mask]

        if noise_mask:
            """
            TODO: Change to this:
                similarity_matrix = MatrixFactorization.apply_IQR_mask(similarity_matrix, percentiles=(0.25,0.75))
            Doesn't need to be applied to both matrices since loss function deals with NaNs
            """

            def get_noise_mask(similarity_matrix: torch.Tensor, lower=None, upper=None):
                if (lower is None) & (upper is None):
                    lower = similarity_matrix.mean() - similarity_matrix.std()
                    upper = similarity_matrix.mean() + similarity_matrix.std()
                mask_greater_than = similarity_matrix > upper
                mask_less_than = similarity_matrix < lower

                return mask_greater_than | mask_less_than

            mask = get_noise_mask(masked_sim_matrix1)
            masked_sim_matrix1 = masked_sim_matrix1[mask]
            masked_sim_matrix2 = masked_sim_matrix2[mask]

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
        noise_mask: bool = False,
        early_stopping: bool = False,
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
            noise_mask (bool): If True then only include extreme similarity values
            early_stopping (bool): Stop when loss function stops reducing (scheduler)
            verbose: If True, print verbose messages during training.

        Returns:
            A tuple containing the trained model, list of losses, and learning rates.
        """

        # Initialize model
        model = MatrixFactorization(
            n_time_series=n_time_series, embedding_dim=embedding_dim, normalize=True
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        PATIENCE = 10
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.8, patience=PATIENCE, threshold=0.1
        )

        # Function to calculate losses
        def get_all_losses(
            model: "MatrixFactorization", target_similarity_matrix: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            pairwise_similarity = model()
            pairwise_loss = pairwise_loss_weight * model.calculate_loss(
                model_similarity_matrix=pairwise_similarity,
                target_similarity_matrix=target_similarity_matrix,
                loss_function=torch.nn.functional.l1_loss,
                noise_mask=noise_mask,
            )
            regularization_loss = (
                regularization_loss_weight
                # * torch.abs(torch.linalg.norm(model.embeddings.weight, dim=1) - 1).sum()
                * torch.square(
                    torch.linalg.norm(model.embeddings.weight, dim=1) - 1
                ).sum()
                # * torch.linalg.norm(model.embeddings.weight, dim=1).sum()
            )
            total_loss = pairwise_loss + regularization_loss
            return total_loss, pairwise_loss, regularization_loss

        # Training loop
        losses: List[Tuple[float, float, float]] = []
        learning_rates: List[float] = []
        # - Keep track of LR for early stopping
        lr_latest = optimizer.param_groups[0]["lr"]
        lr_epoch_changed = [0]
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
            lr = optimizer.param_groups[0]["lr"]
            learning_rates.append(lr)
            # -- If three consecutive patience are hit in lr scheduler then early stop
            if early_stopping:
                if lr < lr_latest:
                    lr_latest = lr
                    lr_epoch_changed.append(epoch)
                    if len(lr_epoch_changed) < 4:
                        pass
                    elif all(np.diff(lr_epoch_changed)[-3:] == [PATIENCE + 1] * 3):
                        print(f"Early stopping at epoch {epoch}")
                        break
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
            title="Losses During Training",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            template="plotly_dark",
            height=300,
            width=600,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.8))

        # Show the figure
        if verbose:
            print(f"Final pairwise_loss: {pairwise_losses[-1]}")
            fig.show()
        if return_fig:
            return fig

    @staticmethod
    def apply_random_mask(similarity_matrix, percentage):
        if similarity_matrix.shape[0] != similarity_matrix.shape[1]:
            raise ValueError("similarity_matrix must be square")

        if not (0 <= percentage <= 1):
            raise ValueError("Percentage must be between 0 and 100")

        # Create a mask to extract non-diagonal elements
        non_diagonal_mask = ~torch.eye(similarity_matrix.shape[0], dtype=torch.bool)

        # Flatten the non-diagonal elements and select a random sample
        non_diagonal_indices = non_diagonal_mask.nonzero(as_tuple=True)
        num_elements_to_mask = int(percentage * non_diagonal_indices[0].shape[0])
        selected_indices = torch.randperm(non_diagonal_indices[0].shape[0])[
            :num_elements_to_mask
        ]

        # Create an initial mask with all False
        mask = torch.zeros_like(similarity_matrix, dtype=torch.bool)

        # Set selected indices to True using advanced indexing
        mask[
            non_diagonal_indices[0][selected_indices],
            non_diagonal_indices[1][selected_indices],
        ] = True

        # Apply mask in-place
        similarity_matrix.masked_fill_(mask, torch.nan)

        return similarity_matrix

    @staticmethod
    def apply_IQR_mask(similarity_matrix, percentiles=(0.25, 0.75)):
        """Get mask to keep only upper and lower values per row
        Args:
            similarity_matrix (_type_): _description_
            percentiles (tuple, optional): Percentiles to determine the threshold for low and high values. Defaults to (0.25, 0.75).
        """
        if similarity_matrix.shape[0] != similarity_matrix.shape[1]:
            raise ValueError("similarity_matrix must be square")

        # Mask out diagonal elements (set them to NaN for now)
        similarity_matrix.fill_diagonal_(torch.nan)

        # Compute the lower and upper quantiles for each row
        # First filter out the diagonals since they are nan and break the rowwise calculation
        n = similarity_matrix.shape[0]
        no_diagonals = (
            similarity_matrix.flatten()[1:].view(n - 1, n + 1)[:, :-1].reshape(n, n - 1)
        )
        lower_quartile = torch.quantile(no_diagonals, percentiles[0], dim=1)
        upper_quartile = torch.quantile(no_diagonals, percentiles[1], dim=1)

        # Expand the quantiles to match the shape of the similarity_matrix
        lower_quartile_expanded = lower_quartile.unsqueeze(1).expand_as(
            similarity_matrix
        )
        upper_quartile_expanded = upper_quartile.unsqueeze(1).expand_as(
            similarity_matrix
        )

        # Create masks based on the quantiles
        lower_mask = similarity_matrix < lower_quartile_expanded
        upper_mask = similarity_matrix > upper_quartile_expanded

        # Apply the masks and set values
        similarity_matrix[lower_mask] = -2
        similarity_matrix[upper_mask] = 2
        similarity_matrix[~(lower_mask | upper_mask)] = torch.nan

        # Restore the diagonal to NaN
        similarity_matrix.fill_diagonal_(torch.nan)

        return similarity_matrix
