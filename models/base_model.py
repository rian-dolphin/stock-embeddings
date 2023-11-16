import numpy as np
import torch
import torch.nn as nn


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

    def calculate_loss(self, output, target):
        # Implement a common method for calculating loss if applicable
        raise NotImplementedError("Loss calculation method not implemented.")
