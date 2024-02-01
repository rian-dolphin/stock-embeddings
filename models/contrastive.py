from collections import defaultdict

import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.base_model import BaseModel
from utils.contrastive_helpers import BaseContrastiveLoss, MultiPosNegDataset


class ContrastiveMultiPN(BaseModel):
    """
    Contrastive learning model for multiple positive samples and multiple negative samples
    """

    def __init__(
        self, n_time_series: int, embedding_dim: int, criterion: BaseContrastiveLoss
    ):
        super(ContrastiveMultiPN, self).__init__()
        self.embeddings = nn.Embedding(n_time_series, embedding_dim)
        self.criterion = criterion
        self.losses = {
            "total": [],
            "learning_rate": [],
            "contrastive": [],
            "regularization": [],
        }

    def train(
        self,
        index_samples,
        batch_size,
        learning_rate,
        epochs,
        regularization_weight=0,
        normalize=False,
    ):
        optimizer = optim.Adam(self.embeddings.parameters(), lr=learning_rate)

        # Create the dataset and data loader
        multi_pos_neg_dataset = MultiPosNegDataset(index_samples)
        data_loader = DataLoader(
            multi_pos_neg_dataset, batch_size=batch_size, shuffle=True
        )

        for epoch in range(epochs):
            epoch_losses = {"total": 0, "contrastive": 0, "regularization": 0}
            # normalise the embeddings to prevent degenerate solution
            if normalize:
                self.embeddings = self.normalize_embeddings(self.embeddings)

            for i, (anchor_idx, positive_idx, negative_indices) in enumerate(
                data_loader
            ):
                batch_loss = torch.zeros(1)
                # Get the embeddings for anchor, positive, and negative
                anchor_embeddings = self.embeddings(
                    anchor_idx
                )  # shape: (batch_size, embedding_dim)
                positive_embeddings = self.embeddings(
                    positive_idx
                )  # shape: (batch_size, num_pos_samples embedding_dim)
                negative_embeddings = self.embeddings(
                    negative_indices
                )  # shape: (batch_size, num_neg_samples, embedding_dim)

                # Compute the loss
                contrastive_loss = self.criterion(
                    anchor_embeddings, positive_embeddings, negative_embeddings
                )
                batch_loss += contrastive_loss
                if regularization_weight > 0:
                    regularization_loss = (
                        regularization_weight
                        * torch.square(self.embeddings.weight.norm(dim=1) - 1).sum()
                    )
                    batch_loss += regularization_loss
                else:
                    regularization_loss = torch.zeros(1)

                # Backward pass and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                # Update epoch loss sums
                epoch_losses["total"] += batch_loss.item()
                epoch_losses["contrastive"] += contrastive_loss.item()
                epoch_losses["regularization"] += regularization_loss.item()

            # Append epoch sums to losses dictionary
            for loss_type, loss_value in epoch_losses.items():
                self.losses[loss_type].append(loss_value / len(data_loader))
            self.losses["learning_rate"].append(optimizer.param_groups[0]["lr"])

            if (epoch % 1 == 0) | (epoch == epochs - 1) | (epoch == 0):
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_losses['total']:.4f}")

    def plot_training(self, plot_lr=True):
        # Create a figure
        fig = go.Figure()

        x_vals = [i + 1 for i in range(len(self.losses["total"]))]

        # Add traces for pairwise and regularization losses
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=self.losses["total"],
                mode="lines",
                name="Total Loss",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=self.losses["contrastive"],
                mode="lines",
                name="Contrastive Loss",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=self.losses["regularization"],
                mode="lines",
                name="Regularization Loss",
            )
        )

        if plot_lr:
            # Create a secondary y-axis for the total loss
            fig.update_layout(
                yaxis=dict(title="Pairwise and Regularization Loss"),
                yaxis2=dict(title="Learning Rate", overlaying="y", side="right"),
            )
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=self.losses["learning_rate"],
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
        fig.show()
