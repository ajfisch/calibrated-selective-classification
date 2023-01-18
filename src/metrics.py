"""Extra metrics."""

import collections
import math

import prettytable
import torch
import torch.nn.functional as F
import numpy as np

from src import utils

# Default minimum coverage level to evaluate.
MIN_COVERAGE = 0.05

# Number of numerical integration points to use for AUC.
NUM_AUC_VALUES = 50

CoverageResult = collections.namedtuple(
    'CoverageResult', ['coverage', 'mean', 'std'], defaults=[None, None, None])

AUCResult = collections.namedtuple(
    'AUCResult', ['auc', 'values'], defaults=[None, None])


def format_metrics(dataset_to_result, metric_names=None):
    """Format test results as PrettyTables.

    Args:
        dataset_to_result: A dict of dataset --> method --> metric --> result.
        metric_names: A list of metrics to include.

    Returns:
        A formatted string.
    """
    metric_names = metric_names or ['ce_2', 'ce_inf', 'brier', 'acc']
    datasets = sorted(dataset_to_result.keys())
    if len(set(datasets) - {'avg'}) == 1:
        datasets = [d for d in datasets if d != 'avg']
    string = []
    for dataset in datasets:
        methods = dataset_to_result[dataset]
        string += [f'\nDataset: {dataset}']
        table = prettytable.PrettyTable()
        table.field_names = ['method'] + list(metric_names)
        for name, results in methods.items():
            row = [name]
            for metric in metric_names:
                result = results[metric]
                if isinstance(result, CoverageResult):
                    row.append(f'{100 * result.mean:2.2f}')
                elif isinstance(result, AUCResult):
                    row.append(f'{100 * result.auc:2.2f}')
                else:
                    raise ValueError('Unknown result type')
            table.add_row(row)
        string.append(table.get_string())
    return '\n'.join(string)


def compute_auc(values):
    """Compute the area under the curve using the trapezoidal method."""
    xs, ys, _ = zip(*values)
    area = np.trapz(ys, xs) / (max(xs) - min(xs) + 1e-8)
    return area


def reduce_mean_coverage(results):
    """List of coverage results."""
    coverage = np.mean([res.coverage for res in results])
    mean = np.mean([res.mean for res in results])
    std = np.std([res.mean for res in results])
    return CoverageResult(coverage, mean, std)


def reduce_mean_auc(results):
    """List of AUCResults."""
    values = []
    for idx in range(len(results[0].values)):
        coverage = np.mean([res.values[idx].coverage for res in results])
        mean = np.mean([res.values[idx].mean for res in results])
        std = np.std([res.values[idx].mean for res in results])
        values.append(CoverageResult(coverage, mean, std))
    return AUCResult(compute_auc(values), values)


def compute_metric_at_coverage(
    outputs,
    targets,
    weights,
    coverage,
    metric_fn,
):
    """Compute given metric function at a specified coverage level.

    Args:
        outputs: Score of f(X) in E[Y | f(X)]. Size [num_examples].
        targets: Target value Y in E[Y | f(X)]. Size [num_examples].
        weights: Soft outputs of \tilde{g}(X). Size [num_examples].
        coverage: Desired selective coverage level.
        metric_fn: Callable for fn(outputs, targets).

    Returns:
       A CoverageResult.
    """
    threshold = utils.calibrate(weights, coverage)
    take = weights.ge(threshold)
    value = metric_fn(outputs[take], targets[take])
    return CoverageResult(coverage, value, None)


def compute_metric_auc(
    outputs,
    targets,
    weights,
    metric_fn,
    min_coverage=MIN_COVERAGE,
    num_auc_values=NUM_AUC_VALUES,
):
    """Compute AUC of the given metric function.

    Args:
        outputs: Score of f(X) in E[Y | f(X)]. Size [num_examples].
        targets: Target value Y in E[Y | f(X)]. Size [num_examples].
        weights: Soft outputs of \tilde{g}(X). Size [num_examples].
        metric_fn: Callable for fn(outputs, targets).
        min_coverage: Minimum coverage level to start from.
        num_auc_values: Number of discrete numerical integration points.

    Returns:
       An AUCResult containing the auc and a list of CoverageResults.
    """
    values = []
    for coverage in np.linspace(min_coverage, 1, num_auc_values):
        values.append(compute_metric_at_coverage(
            outputs, targets, weights, coverage, metric_fn))
    return AUCResult(compute_auc(values), values)


def compute_accuracy(outputs, targets, is_binary=False):
    """Compute accuracy of the predictions.

    The outputs and targets are either assumed to be

    (1) A binary label Y and the prediction score p(Y = 1 | X). In this case,
        the simple classification rule \hat{Y} = 1{p(Y = 1 | X) ≥ 0.5} is
        applied, and the accuracy is computed by comparing \hat{Y} to Y.

    (2) The original task is a multi-class prediction problem, and targets
        already represents the "correctness" reduction, Y = argmax p(y | X).
        In this case, the accuracy is simple the average target value.

    Args:
        outputs: Score of f(X) in E[Y | f(X)]. Size [num_selected].
        targets: Target value Y in E[Y | f(X)]. Size [num_selected].
        is_binary: Whether to treat Y as an original binary label.

    Returns:
        The model accuracy.
    """
    if len(outputs) == 0:
        return 1

    if is_binary:
        # Outputs = f(X), and targets = binary Y.
        acc = outputs.ge(0.5).eq(targets).float().mean()
    else:
        # Outputs = max f(y | x) and targets = correctness.
        acc = targets.float().mean()
    return acc.item()


def compute_brier_score(outputs, targets):
    """Compute the Brier Score, E[(Y - f(X))^2].

    Args:
        outputs: Score of f(X) in E[Y | f(X)]. Size [num_selected].
        targets: Target value Y in E[Y | f(X)]. Size [num_selected].

    Returns:
        The Brier score.
    """
    if len(outputs) == 0:
        return 0

    return F.mse_loss(outputs.view(-1), targets.view(-1)).item()


def compute_ece(outputs, targets, num_bins=15, min_bin_size=50, pnorm=2):
    """Helper for computing ECE.

    Args:
        outputs: Score of f(X) in E[Y | f(X)]. Size [num_selected].
        targets: Target value Y in E[Y | f(X)]. Size [num_selected].
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
