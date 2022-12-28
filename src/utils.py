"""Random functions."""

import math
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim


def format_train_metrics(iteration, total_iterations, loss, smmce, coverage):
    """Format training log step.

    Args:
        iteration: The current step.
        total_iterations: Total number of steps in the epoch.
        loss: The aggregate loss value.
        smmce: The raw value of the S-MMCE_U component.
        coverage: The raw value of the coverage component.

    Returns:
        A formatted string.
    """
    string = [f"iter = {iteration:04d}/{total_iterations:04d}"]
    string.append("total loss " + f"{loss:.3e}".ljust(6))
    string.append("smmce loss " + f"{smmce:.3e}".ljust(6))
    string.append("coverage loss " + f"{coverage:.3e}".ljust(6))
    return "   ".join(string)


def format_validation_metrics(results):
    """Format validation results.

    Args:
        results: List of AUC results for all datasets.

    Returns:
        A formatted string.
    """
    string = ['AUC']
    string.append(f'Mean = {100 * np.mean(results):.3e}')
    string.append(f'Max = {100 * np.max(results):.3e}')
    string.append(f'p10 = {100 * np.quantile(results, .1):.3e}')
    string.append(f'p90 = {100 * np.quantile(results, .9):.3e}')
    return " | ".join(string)


def temperature_scale(logits, targets):
    """Compute temperature calibration for a network.

    Args:
        logits: Tensor of size [num_examples, num_classes].
        targets: Tensor of size [num_examples] with target class index.

    Returns:
        Tuned temperature value.
    """
    # Optimize temperature.
    temperature = torch.tensor(1.5, requires_grad=True, device=logits.device)
    optimizer = optim.LBFGS([temperature], lr=0.1, max_iter=50)

    def eval():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / temperature, targets)
        loss.backward()
        return loss
    optimizer.step(eval)

    return temperature.item()


def platt_scale(logits, targets):
    """Compute Platt scaling for a network.

    Args:
        logits: Tensor of size [num_examples].
        targets: Tensor of size [num_examples] with binary target 0/1.

    Returns:
        Tuned weight and bias in sigmoid(weight * logit + bias).

    """
    # Optimize temperature.
    weight = torch.tensor(1.0, requires_grad=True, device=logits.device)
    bias = torch.tensor(0.0, requires_grad=True, device=logits.device)
    optimizer = optim.LBFGS([weight, bias], lr=0.1, max_iter=50)

    def eval():
        optimizer.zero_grad()
        scaled_logits = weight * logits + bias
        loss = F.binary_cross_entropy_with_logits(scaled_logits, targets)
        loss.backward()
        return loss
    optimizer.step(eval)

    return weight.item(), bias.item()


def calibrate(weights, coverage):
    """Return coverage preserving threshold over weights.

    Args:
        weights: Tensor of size [num_examples] raw (soft) \tilde(g)(X) values.
        coverage: Target coverage level in [0, 1].

    Returns:
        Threshold for binarizing g(X) in 1{\tilde{g}(X) ≥ threshold}.
    """
    if coverage == 0:
        return float('inf')
    values = -torch.sort(-weights.view(-1)).values
    index = max(0, math.ceil(coverage * len(values)) - 1)
    threshold = values[index].item()
    return threshold


def check_nonempty(dirname):
    """Check if a directory is non-empty."""
    if not os.path.exists(dirname):
        return False
    if not os.path.isdir(dirname):
        raise ValueError(f'{dirname} is not a directory.')
    if len(os.listdir(dirname)) > 0:
        return True
    return False
