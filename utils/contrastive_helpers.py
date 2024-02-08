import torch
import torch.nn as nn
from torch.utils.data import Dataset


class MultiPosNegDataset(Dataset):
    def __init__(self, index_samples):
        self.index_samples = index_samples

    def __len__(self):
        return len(self.index_samples)

    def __getitem__(self, idx):
        anchor_idx, positive_indices, negative_indices = self.index_samples[idx]
        positive_indices_tensor = torch.tensor(positive_indices)
        negative_indices_tensor = torch.tensor(negative_indices)
        return anchor_idx, positive_indices_tensor, negative_indices_tensor


class BaseContrastiveLoss(nn.Module):
    def __init__(self, positive_weight=1, negative_weight=1):
        super(BaseContrastiveLoss, self).__init__()
        self.positive_weight = positive_weight
        self.negative_weight = negative_weight
        self.positive_negative_ratio = None

    def forward(
        self,
        anchor_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
        negative_embeddings: torch.Tensor,
    ):
        """
        Args:
            anchor_embeddings (torch.Tensor): shape (batch_size, embedding_dim)
            positive_embeddings (torch.Tensor): (batch_size, num_pos_samples, embedding_dim)
            negative_embeddings (torch.Tensor): (batch_size, num_neg_samples, embedding_dim)

        Returns:
            torch.Tensor: Containing single loss value
        """
        raise NotImplementedError("This method should be overridden by subclasses.")


class IndividualSigmoidLoss(BaseContrastiveLoss):
    def __init__(self, positive_weight=1, negative_weight=1):
        super(IndividualSigmoidLoss, self).__init__()
        self.positive_weight = positive_weight
        self.negative_weight = negative_weight
        self.BCELogitsCriterion = torch.nn.BCEWithLogitsLoss()

    def forward(
        self, anchor_embeddings, positive_embeddings, negative_embeddings
    ) -> torch.Tensor:
        positive_scores = torch.einsum(
            "bpd,bd->bp", [positive_embeddings, anchor_embeddings]
        )
        negative_scores = torch.einsum(
            "bnd,bd->bn", [negative_embeddings, anchor_embeddings]
        )
        # positive_loss = - torch.sum(torch.nn.functional.logsigmoid(positive_scores), dim=1)
        # negative_loss = - torch.sum(torch.log(1-torch.sigmoid(negative_scores)), dim=1)
        positive_loss = self.BCELogitsCriterion(
            positive_scores, torch.ones_like(positive_scores)
        )
        negative_loss = self.BCELogitsCriterion(
            negative_scores, torch.zeros_like(negative_scores)
        )
        loss = (
            self.positive_weight * positive_loss + self.negative_weight * negative_loss
        )
        self.positive_negative_ratio = positive_loss / negative_loss

        return loss


class AggregateSigmoidLoss(BaseContrastiveLoss):
    def __init__(self, positive_weight=1, negative_weight=1):
        super(AggregateSigmoidLoss, self).__init__()
        self.positive_weight = positive_weight
        self.negative_weight = negative_weight
        self.BCELogitsCriterion = torch.nn.BCEWithLogitsLoss()

    def forward(
        self, anchor_embeddings, positive_embeddings, negative_embeddings
    ) -> torch.Tensor:
        positive_scores = torch.einsum(
            "bd,bd->b", [torch.mean(positive_embeddings, dim=1), anchor_embeddings]
        )
        negative_scores = torch.einsum(
            "bd,bd->b", [torch.mean(negative_embeddings, dim=1), anchor_embeddings]
        )
        # positive_loss = - torch.sum(torch.nn.functional.logsigmoid(positive_scores))
        # negative_loss = - torch.sum(torch.log(1-torch.sigmoid(negative_scores)+0.00001))
        positive_loss = self.BCELogitsCriterion(
            positive_scores, torch.ones_like(positive_scores)
        )
        negative_loss = self.BCELogitsCriterion(
            negative_scores, torch.zeros_like(negative_scores)
        )

        loss = (
            self.positive_weight * positive_loss + self.negative_weight * negative_loss
        )
        self.positive_negative_ratio = positive_loss / negative_loss

        return loss


class IndPos_AggSoftmax(BaseContrastiveLoss):
    """Individual Sigmoid Positive with Aggregate Softmax"""

    def __init__(self, positive_weight=1, negative_weight=1):
        super(IndPos_AggSoftmax, self).__init__()
        self.positive_weight = positive_weight
        self.negative_weight = negative_weight
        self.BCELogitsCriterion = torch.nn.BCEWithLogitsLoss()
        self.NLLCriterion = torch.nn.NLLLoss()

    def forward(
        self, anchor_embeddings, positive_embeddings, negative_embeddings
    ) -> torch.Tensor:
        ### POSITIVE LOSS - Sigmoid over each positive
        positive_scores = torch.einsum(
            "bpd,bd->bp", [positive_embeddings, anchor_embeddings]
        )
        positive_loss = self.BCELogitsCriterion(
            positive_scores, torch.ones_like(positive_scores)
        )

        ### NEGATIVE LOSS (and some positive) - Softmax with aggregate positive representation
        aggregate_positive_embeddings = torch.mean(
            positive_embeddings, dim=1
        )  # (batch_size, embedding_dim)
        aggregate_positive_scores = torch.einsum(
            "bd,bd->b", [aggregate_positive_embeddings, anchor_embeddings]
        )
        negative_scores = torch.einsum(
            "bnd,bd->bn", [negative_embeddings, anchor_embeddings]
        )
        # -- Make first column the aggregate positive scores
        # - concatenated.shape = (batch_size, 1+num_neg_samples)
        concatenated = torch.concat(
            (aggregate_positive_scores.unsqueeze(1), negative_scores), dim=1
        )
        # -- Create target for NLLLoss
        # zeros indicate the first element of concatenated is the target class, which corresponds to the positive sample
        target = torch.zeros(concatenated.shape[0], dtype=torch.long)

        negative_loss = self.NLLCriterion(
            torch.nn.functional.log_softmax(concatenated, dim=1), target
        )
        loss = (
            self.positive_weight * positive_loss + self.negative_weight * negative_loss
        )
        self.positive_negative_ratio = positive_loss / negative_loss

        return loss


class AggPos_IndNeg(BaseContrastiveLoss):
    """Aggregate Positive with Individual Sigmoid for negatives"""

    def __init__(self, positive_weight=1, negative_weight=1):
        super(AggPos_IndNeg, self).__init__()
        self.positive_weight = positive_weight
        self.negative_weight = negative_weight
        self.BCELogitsCriterion = torch.nn.BCEWithLogitsLoss()

    def forward(
        self, anchor_embeddings, positive_embeddings, negative_embeddings
    ) -> torch.Tensor:
        ### POSITIVE LOSS - done with aggregation
        positive_scores = torch.einsum(
            "bd,bd->b", [torch.mean(positive_embeddings, dim=1), anchor_embeddings]
        )

        positive_loss = self.BCELogitsCriterion(
            positive_scores, torch.ones_like(positive_scores)
        )

        ### NEGATIVE LOSS - Sigmoid over each negative
        negative_scores = torch.einsum(
            "bnd,bd->bn", [negative_embeddings, anchor_embeddings]
        )
        negative_loss = self.BCELogitsCriterion(
            negative_scores, torch.zeros_like(negative_scores)
        )

        loss = (
            self.positive_weight * positive_loss + self.negative_weight * negative_loss
        )
        self.positive_negative_ratio = positive_loss / negative_loss

        return loss
