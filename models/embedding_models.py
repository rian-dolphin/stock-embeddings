import torch
import torch.nn as nn
import torch.nn.functional as F

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

        # Ensure that the input matrices are square and of the same size
        assert (
            similarity_matrix1.shape == similarity_matrix2.shape
        ), "Matrices must be of the same size."
        assert (
            similarity_matrix1.shape[0] == similarity_matrix1.shape[1]
        ), "Matrices must be square."

        # Create a mask for non-diagonal elements
        n = similarity_matrix1.size(0)
        mask = torch.eye(n, dtype=torch.bool).logical_not()

        # Apply the mask
        masked_sim_matrix1 = similarity_matrix1[mask]
        masked_sim_matrix2 = similarity_matrix2[mask]

        # Calculate the loss using the provided loss function
        loss_value = loss_function(
            masked_sim_matrix1, masked_sim_matrix2, reduction="sum"
        )

        return loss_value
