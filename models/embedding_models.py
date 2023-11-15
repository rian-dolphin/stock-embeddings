from models.base_model import BaseModel
import torch.nn as nn
import torch


class ClassificationEmbeddings(BaseModel):
    """
    Model architecture similar to CBOW Word2Vec but adapted for stock modelling.
    """

    def __init__(self, n_time_series: int, embedding_dim: int):
        super(ClassificationEmbeddings, self).__init__()
        self.embeddings = nn.Embedding(n_time_series, embedding_dim)

    def forward(self, inputs):
        # -- This extracts the relevant rows of the embedding matrix
        # - Equivalent to W^T x_i in "word2vec Parameter Learning Explained"
        temp = self.embeddings(inputs)  # .view((len(inputs),-1))

        # -- Compute the hidden layer by a simple mean
        hidden = temp.mean(axis=1)
        # -- Reshape to make matrix dimensions compatible
        hidden = hidden.unsqueeze(dim=2)
        # -- Compute dot product of hidden with embeddings
        # out = torch.einsum("nd,", self.embeddings.weight, hidden)
        out = torch.matmul(self.embeddings.weight, hidden)

        # -- Return the log softmax since we use NLLLoss loss function
        return nn.functional.log_softmax(out, dim=1)
