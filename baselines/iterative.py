"""Small EDDI scoring helpers retained for the online implementation."""

import numpy as np
import torch


def valid_probs(predictions):
    """Return whether every prediction is in the probability interval."""
    return torch.all((predictions >= 0) & (predictions <= 1))


def calculate_criterion(predictions, task):
    """Calculate EDDI's prediction-variability selection criterion."""
    if task == "regression":
        return torch.var(predictions)
    if task != "classification":
        raise ValueError("task must be classification or regression")

    if len(predictions.shape) == 1 or predictions.shape[1] == 1:
        if not valid_probs(predictions):
            predictions = predictions.sigmoid()
        if len(predictions.shape) == 1:
            predictions = predictions.view(-1, 1)
        predictions = torch.cat([1 - predictions, predictions])
    elif not valid_probs(predictions):
        predictions = predictions.softmax(dim=1)

    mean = torch.mean(predictions, dim=0, keepdim=True)
    kl = torch.sum(
        predictions * torch.log(predictions / (mean + 1e-6) + 1e-6), dim=1
    )
    return torch.mean(kl)


class Imputer:
    """Write sampled values into one feature or modality group."""

    def __init__(self, group_matrix=None):
        self.group_matrix = group_matrix

    def impute(self, x, x_ind, index):
        if self.group_matrix is not None:
            index = np.where(self.group_matrix[index] == 1)[0]
        x[:, index] = x_ind
        return x
