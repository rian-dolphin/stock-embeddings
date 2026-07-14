import torch
import torch.nn as nn


class BaseContrastiveLoss(nn.Module):
    def __init__(self, positive_weight=1, negative_weight=1):
        super(BaseContrastiveLoss, self).__init__()
        self.positive_weight = positive_weight
        self.negative_weight = negative_weight

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

    def update_loss_weights(
        self, positive_loss, negative_loss, target_ratio=1, target_magnitude=5
    ):
        """Overwrites the weights to make the positive and negative loss have the same magnitude

        Args:
            positive_loss: positive loss value for epoch
            negative_loss: positive loss value for epoch
            target_ratio (int, optional): . Defaults to 1.
            target_magnitude (int, optional): Desired pos+neg loss magnitude. Defaults to 5.
        """
        if target_ratio != 1:
            raise NotImplementedError("target ratio not implemented, will always be 1")
        loss_ratio = positive_loss / negative_loss
        magnitude_factor = target_magnitude / (2 * negative_loss)
        self.positive_weight = (
            self.positive_weight * (1 / loss_ratio) * magnitude_factor
        )
        self.negative_weight = self.negative_weight * magnitude_factor


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
        positive_loss = self.positive_weight * self.BCELogitsCriterion(
            positive_scores, torch.ones_like(positive_scores)
        )
        negative_loss = self.negative_weight * self.BCELogitsCriterion(
            negative_scores, torch.zeros_like(negative_scores)
        )
        loss = positive_loss + negative_loss

        return loss, positive_loss, negative_loss


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
        positive_loss = self.positive_weight * self.BCELogitsCriterion(
            positive_scores, torch.ones_like(positive_scores)
        )
        negative_loss = self.negative_weight * self.BCELogitsCriterion(
            negative_scores, torch.zeros_like(negative_scores)
        )

        loss = positive_loss + negative_loss

        return loss, positive_loss, negative_loss


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
        positive_loss = self.positive_weight * self.BCELogitsCriterion(
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

        negative_loss = self.negative_weight * self.NLLCriterion(
            torch.nn.functional.log_softmax(concatenated, dim=1), target
        )
        loss = positive_loss + negative_loss

        return loss, positive_loss, negative_loss


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

        positive_loss = self.positive_weight * self.BCELogitsCriterion(
            positive_scores, torch.ones_like(positive_scores)
        )

        ### NEGATIVE LOSS - Sigmoid over each negative
        negative_scores = torch.einsum(
            "bnd,bd->bn", [negative_embeddings, anchor_embeddings]
        )
        negative_loss = self.negative_weight * self.BCELogitsCriterion(
            negative_scores, torch.zeros_like(negative_scores)
        )

        loss = positive_loss + negative_loss

        return loss, positive_loss, negative_loss


class IndPos_AggNeg(BaseContrastiveLoss):
    def __init__(self, positive_weight=1, negative_weight=1):
        super(IndPos_AggNeg, self).__init__()
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
            "bd,bd->b", [torch.mean(negative_embeddings, dim=1), anchor_embeddings]
        )

        positive_loss = self.positive_weight * self.BCELogitsCriterion(
            positive_scores, torch.ones_like(positive_scores)
        )
        negative_loss = self.negative_weight * self.BCELogitsCriterion(
            negative_scores, torch.zeros_like(negative_scores)
        )

        loss = positive_loss + negative_loss

        return loss, positive_loss, negative_loss


class JointPos_MarginalNeg(BaseContrastiveLoss):
    def __init__(self, positive_weight=1, negative_weight=1):
        super(JointPos_MarginalNeg, self).__init__()
        self.positive_weight = positive_weight
        self.negative_weight = negative_weight
        self.BCELogitsCriterion = torch.nn.BCEWithLogitsLoss()

    def forward(
        self, anchor_embeddings, positive_embeddings, negative_embeddings
    ) -> torch.Tensor:
        pos_joint_probs_pre_sigmoid = torch.einsum(
            "bpd,bd->bp", positive_embeddings, anchor_embeddings
        )
        # -- Need to do something here to prevent losses becoming huge
        # - Product of multiple probs becomes very small
        # pos_joint_probs_pre_sigmoid = torch.prod(pos_joint_probs_pre_sigmoid, dim=1)
        pos_joint_probs_pre_sigmoid = torch.mean(pos_joint_probs_pre_sigmoid, dim=1)
        # pos_joint_probs_pre_sigmoid = torch.min(
        #     pos_joint_probs_pre_sigmoid, dim=1
        # ).values
        neg_marginal_probs_pre_sigmoid = torch.einsum(
            "bnd,bd->bn", negative_embeddings, anchor_embeddings
        )

        pos_joint_probs_pre_sigmoid = pos_joint_probs_pre_sigmoid.flatten()
        neg_marginal_probs_pre_sigmoid = neg_marginal_probs_pre_sigmoid.flatten()

        positive_loss = self.positive_weight * self.BCELogitsCriterion(
            pos_joint_probs_pre_sigmoid, torch.ones_like(pos_joint_probs_pre_sigmoid)
        )
        negative_loss = self.negative_weight * self.BCELogitsCriterion(
            neg_marginal_probs_pre_sigmoid,
            torch.zeros_like(neg_marginal_probs_pre_sigmoid),
        )

        loss = positive_loss + negative_loss
        return loss, positive_loss, negative_loss
