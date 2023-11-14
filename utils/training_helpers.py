import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from models.base_model import BaseModel

# Constants
LEARNING_RATE = 0.0001
LOSS_THRESHOLD = 0.0002


def validate_params(model, embedding_dim, idx_combinations):
    if idx_combinations is None:
        raise ValueError("idx_combinations must be provided for training.")

    if model is None and embedding_dim is None:
        raise ValueError("If no model is provided, embedding_dim must be specified.")

    if model and embedding_dim:
        raise ValueError("Do not provide embedding_dim when a model is given.")


def initialize_model(model_class, n_time_series, embedding_dim):
    return model_class(n_time_series, embedding_dim)


def get_train_loader(idx_combinations, batch_size=64):
    x_vals = np.array([idx[1] for idx in idx_combinations])
    y_vals = np.array([idx[0] for idx in idx_combinations])

    x_train_tensor = torch.from_numpy(x_vals).long()
    y_train_tensor = torch.from_numpy(y_vals).long()

    train_data = CustomDataset(x_train_tensor, y_train_tensor)
    return DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)


def should_early_stop(losses, epoch):
    return epoch > 2 and abs(losses[-1] - losses[-2]) < LOSS_THRESHOLD


def train_embeddings_from_idx_combinations(
    n_time_series,
    idx_combinations=None,
    model=None,
    epochs=20,
    embedding_dim=None,
    device="cpu",
    verbose=True,
):
    validate_params(model, embedding_dim, idx_combinations)

    if model is None:
        model = initialize_model(BaseModel, n_time_series, embedding_dim)
    else:
        embedding_dim = model.embeddings.weight.shape[1]

    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_loader = get_train_loader(idx_combinations)

    if verbose:
        print("Training embeddings...")

    losses = train_model(
        epochs, model, loss_function, optimizer, train_loader, device, verbose
    )

    return losses


def train_model(
    epochs: int,
    model: nn.Module,
    loss_function: nn.Module,
    optimizer: optim.Optimizer,
    train_loader: DataLoader,
    device: str = "cpu",
    verbose: bool = True,
) -> list:
    """
    Train a PyTorch model.

    Parameters
    ----------
    epochs : int
        The number of epochs for training.
    model : nn.Module
        The PyTorch model to be trained.
    loss_function : nn.Module
        The loss function for training.
    optimizer : optim.Optimizer
        The optimizer for training.
    train_loader : DataLoader
        The data loader for training data.
    device : str, optional
        The device ('cuda', 'cpu', etc.) for training.
    verbose : bool, optional
        Flag to control verbosity of training process.

    Returns
    -------
    list
        A list containing the loss values for each epoch.
    """
    model.to(device)
    losses = []

    for epoch in tqdm(range(epochs), disable=not verbose):
        total_loss = 0
        count = 0

        for x_batch, y_batch in train_loader:
            count += len(y_batch)
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            log_probs = model(x_batch)
            loss = loss_function(log_probs.squeeze(), y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        epoch_loss = total_loss / count
        losses.append(epoch_loss)

        if verbose:
            print(f"Epoch {epoch}: Loss = {epoch_loss}")

        if should_early_stop(losses, epoch):
            if verbose:
                print(f"Early stopping at epoch {epoch} due to minimal loss reduction.")
            break

    return model, losses


class CustomDataset(Dataset):
    def __init__(self, x_tensor: torch.Tensor, y_tensor: torch.Tensor):
        self.x = x_tensor
        self.y = y_tensor

    def __getitem__(self, index: int) -> tuple:
        return self.x[index], self.y[index]

    def __len__(self) -> int:
        return len(self.x)
