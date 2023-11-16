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

    def initialize_parameters(self, method="xavier_uniform"):
        if method == "xavier_uniform":
            nn.init.xavier_uniform_(self.embeddings.weight)
        # Add other initialization methods as needed

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device))

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

    def validate(self, validation_loader):
        # Implement validation logic
        pass

    # Additional methods like logging, regularization etc.
    # ...
