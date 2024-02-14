import time
from collections import defaultdict

import numpy as np
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
            "positive": [],
            "negative": [],
            "regularization": [],
        }
        self.lr_latest = None
        self.lr_epoch_changed = [0]

    def train(
        self,
        index_samples,
        batch_size,
        learning_rate,
        epochs,
        regularization_weight=0,
        patience=10,
        normalize=False,
        early_stopping: bool = False,
        print_every=1,
    ):
        if self.lr_latest is None:
            optimizer = optim.Adam(self.embeddings.parameters(), lr=learning_rate)
        else:
            if learning_rate != self.lr_latest:
                print(f"Resuming training using last learning rate: {self.lr_latest}")
            optimizer = optim.Adam(self.embeddings.parameters(), lr=self.lr_latest)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.8, patience=patience, threshold=0.1
        )
        if self.lr_latest is None:
            self.lr_latest = optimizer.param_groups[0]["lr"]

        # Create the dataset and data loader
        multi_pos_neg_dataset = MultiPosNegDataset(index_samples)
        data_loader = DataLoader(
            multi_pos_neg_dataset, batch_size=batch_size, shuffle=True
        )
        start_time = time.time()
        for epoch in range(epochs):
            epoch_start_time = time.time()
            epoch_losses = {k: 0 for k in self.losses.keys()}
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
                contrastive_loss, positive_loss, negative_loss = self.criterion(
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
                epoch_losses["total"] += batch_loss
                epoch_losses["contrastive"] += contrastive_loss
                epoch_losses["positive"] += positive_loss
                epoch_losses["negative"] += negative_loss
                epoch_losses["regularization"] += regularization_loss

            # Append epoch sums to losses dictionary
            for loss_type, loss_value in epoch_losses.items():
                if loss_type == "learning_rate":
                    continue
                self.losses[loss_type].append(loss_value.item() / len(data_loader))

            lr = optimizer.param_groups[0]["lr"]
            self.losses["learning_rate"].append(lr)

            scheduler.step(epoch_losses["contrastive"])

            # -- If three consecutive patience are hit in lr scheduler then early stop
            if early_stopping:
                if lr < self.lr_latest:
                    self.lr_latest = lr
                    self.lr_epoch_changed.append(epoch)
                    if len(self.lr_epoch_changed) < 4:
                        pass
                    elif all(np.diff(self.lr_epoch_changed)[-3:] == [patience + 1] * 3):
                        print(f"Early stopping at epoch {epoch}")
                        break

            # Print updates
            epoch_end_time = time.time()  # End time of the epoch
            epoch_duration = epoch_end_time - epoch_start_time
            total_time_elapsed = epoch_end_time - start_time
            estimated_time_to_completion = (total_time_elapsed / (epoch + 1)) * (
                epochs - epoch - 1
            )
            if (epoch % print_every == 0) | (epoch == epochs - 1) | (epoch == 0):
                update_string = (
                    f"=== Epoch [{epoch+1}/{epochs}] ===\n"
                    f"Contrastive Loss: {self.losses['contrastive'][-1]:.4f}  |  "
                    f"Total Loss: {self.losses['total'][-1]:.4f},\n"
                    f"Positive Loss: {self.losses['positive'][-1]:.4f}  |  "
                    f"Negative Loss: {self.losses['negative'][-1]:.4f},\n"
                    f"Epoch Time: {epoch_duration:.2f} sec |  "
                    f"Remaining: {estimated_time_to_completion:.2f} sec,\n"
                )
                print(update_string)

    def plot_training(self, skip=0, plot_lr=True):
        # Create a figure
        fig = go.Figure()

        x_vals = [i + 1 for i in range(len(self.losses["total"]))]

        for loss_name, loss_values in self.losses.items():
            if loss_name == "learning_rate":
                continue
            if not all(np.array(loss_values) == 0):
                fig.add_trace(
                    go.Scatter(
                        x=x_vals[skip:],
                        y=loss_values[skip:],
                        mode="lines",
                        name=loss_name,
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
                    x=x_vals[skip:],
                    y=self.losses["learning_rate"][skip:],
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
