"""Extra metrics."""

import math
import collections

import torch
import torch.nn.functional as F
import numpy as np

from src import utils

NUM_AUC_VALUES = 50

AUC = collections.namedtuple('AUC', ['auc', 'values'])


def compute_auc(values):
    """Compute the area under the curve using the trapezoidal method."""
    xs, ys = zip(*values)
    area = np.trapz(ys, xs) / (max(xs) - min(xs) + 1e-8)
    return area


def compute_metric_auc(
    outputs,
    targets,
    weights,
    metric_fn,
    num_auc_values=NUM_AUC_VALUES,
):
    """Compute AUC of the given metric function."""
    values = []
    for coverage in np.linspace(0, 1, num_auc_values):
        threshold = utils.calibrate(weights, coverage)
        take = weights.ge(threshold)
        metric = metric_fn(outputs[take], targets[take])
        values.append((coverage, metric))
    return AUC(compute_auc(values), values)


def compute_accuracy(outputs, targets, is_binary=True):
    """Compute accuracy of outputs."""
    if is_binary:
        # Outputs = f(X), and targets = binary Y.
        acc = outputs.ge(0.5).eq(targets).mean()
    else:
        # Outputs = max f(y | x) and targets = correctness.
        acc = targets.mean()
    return acc.item()


def compute_brier_score(outputs, targets):
    """Compute the Brier Score, E[(Y - f(X))^2]."""
    return F.mse_loss(outputs.view(-1), targets.view(-1)).item()


def compute_ece(outputs, targets, num_bins=15, min_bin_size=50, pnorm=2):
    """Helper for computing ECE.

    Args:
        outputs: Score of f(X) in E[Y | f(X)].
        targets: Target value Y in E[Y | f(X)].
        num_bins: Largest number of bins to use when computing ECE.
        min_bin_size: Minumum bin size when dividing data.
        pnorm: Value of p to use when computing l_p norm.

    Returns:
        Value of the binned ECE estimator.
    """
    if len(outputs) == 0:
        return 0

    # Sort values.
    num_examples = outputs.numel()
    num_bins = max(min(num_bins, num_examples // min_bin_size), 1)
    outputs, indices = torch.sort(outputs.float())
    targets = torch.index_select(targets.float(), 0, indices)

    # Compute equal-mass bins with average outputs.
    bin_values = torch.tensor_split(outputs, num_bins)
    bin_outputs = [torch.mean(values) for values in bin_values]
    bin_weights = [len(values) / num_examples for values in bin_values]

    # Compute average value of Y per bin.
    bin_labels = torch.tensor_split(targets, num_bins)
    bin_acc = [torch.mean(values) for values in bin_labels]

    # Compute l_p ECE.
    if math.isinf(pnorm):
        ece = torch.tensor(0.0).to(outputs.device)
        for i in range(num_bins):
            ece = max(ece, torch.abs(bin_outputs[i] - bin_acc[i]))
    else:
        ece = torch.tensor(0.0).to(outputs.device)
        for i in range(num_bins):
            bin_ce = torch.abs(bin_outputs[i] - bin_acc[i]).pow(pnorm)
            ece += bin_weights[i] * bin_ce
        ece = ece.pow(1 / pnorm)

    return ece.item()
