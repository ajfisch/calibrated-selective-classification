"""Implementation of S-MMCE losses."""

import torch


def laplacian_kernel(values_i, values_j, width=0.2):
    """Compute the Laplace kernel k(_i, v_j) = exp(-|v_i - v_j| / width).

    Args:
        values_i: n-d matrix of "i" elements.
        values_j: n-d matrix of "j" elements.
        width: Laplacian kernel width parameter.

    Returns:
        n-d matrix of kernel values.
    """
    pairwise_dists = torch.abs(values_i - values_j)
    return torch.exp(-pairwise_dists.div(width))


def compute_smmce_loss(outputs, targets, weights, kernel_fn=None, pnorm=2):
    """Compute the non-normalized S-MMCE_u loss.

    \sum_{ij} |y_i - r_i|^q |y_j - r_j|^q g(x_i) g(x_j) k(r_i, r_j)

    Args:
        outputs: Confidence values r_i.
        targets: Target values y_i.
        weights: Selection weights g(x_i).
        kernel_fn: Callable function to compute k(r_i, r_j) over a matrix.
        pnorm: l_p parameter.

    Returns:
        Computed loss.
    """
    # |y - f(x)|^p
    calibration_error = torch.abs(targets.float() - outputs).pow(pnorm)

    # |x_i - f(x_i)|^p * |y_j - f(x_j)|^p
    pairwise_errors = torch.outer(calibration_error, calibration_error)

    # k(f(x_i), f(x_j))
    kernel_fn = kernel_fn if kernel_fn is not None else laplacian_kernel
    outputs_i = outputs.view(-1, 1).repeat(1, len(outputs))
    outputs_j = outputs.view(1, -1).repeat(len(outputs), 1)
    kernel_matrix = kernel_fn(outputs_i, outputs_j)

    # g(x_i) * g(x_j)
    weights = torch.outer(weights, weights)

    # Compute full matrix: error_i * error_j * g_i * g_j * k(i, j).
    matrix_values = pairwise_errors * kernel_matrix * weights

    # Here we *do not* do any normalization, as described in Eq. 15.
    smmce_loss = matrix_values.sum().pow(1 / pnorm)

    return smmce_loss


def compute_log_regularization_loss(weights):
    """Compute log regularization term to avoid weights collapse to 0.

    \sum_i -log(g(x_i))

    Args:
        weights: Selection weights g(x_i).

    Returns:
        Computed loss.
    """
    weights = torch.clamp(weights, min=1e-8, max=1)
    regularization_loss = -torch.log(weights).sum()
    return regularization_loss
