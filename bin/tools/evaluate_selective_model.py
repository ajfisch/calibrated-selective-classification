"""Evaluate selector."""

import argparse
import collections
import os
import functools

import numpy as np
import torch
import tqdm

from src import metrics
from src import models
from src.data import BatchedInputDataset, InputDataset

parser = argparse.ArgumentParser()

parser.add_argument(
    '--model-files', type=str, nargs='+',
    help='Paths to model files (used for mean + std).')

parser.add_argument(
    '--datasets', type=str, nargs='+',
    help='Paths to datasets to evaluate.')

parser.add_argument(
    '--output-file', type=str,
    help='Path to directory where results will be saved.')

parser.add_argument(
    '--bootstraps', type=int, default=5,
    help='Number of resamples from each test dataset (used for mean + std).')

parser.add_argument(
    '--coverage-level', type=float, default=-1,
    help='Target coverage level (or AUC if ≤ 0).')

parser.add_argument(
    '--batch-size', type=int, default=128,
    help='Batch size for evaluating datasets.')

parser.add_argument(
    '--use-cpu', action='store_true',
    help='Force to run only on CPU.')


# Map of name to function for metrics we compute.
METRICS = {
    # The selective l_2 calibration error.
    'ce_2': functools.partial(metrics.compute_ece, pnorm=2),
    # The selective l_inf calibration error.
    'ce_inf': functools.partial(metrics.compute_ece, pnorm=float('inf')),
    # The selective Brier score.
    'brier': metrics.compute_brier_score,
    # The selective accuracy.
    'acc': metrics.compute_accuracy,
}


def evaluate_model(selector, data_loader, coverage=-1):
    """Evaluate the selective calibration error AUC of the selector.

    Args:
        selector: SelectiveNet nn.Module implementing the selector g(X).
        data_loader: DataLoader for evaluation data, where each batch can be
            converted into either a BatchedInputDataset or a InputDataset.
        coverage: Desired coverage level (-1 = AUC).

    Returns:
        A Result namedtuple with the mean metrics computed (the mean is only
        taken if the input is an instance of a BatchedInputDataset).
    """
    selector.eval()
    device = next(selector.parameters()).device
    all_outputs, all_targets, all_weights = [], [], []

    # Infer from the first batch if the input is a BatchedInputDataset or not.
    is_batched = False

    with torch.no_grad():
        for batch in data_loader:
            if batch[0].dim() == 2:
                batch = InputDataset(*[ex.to(device) for ex in batch])
            else:
                is_batched = True
                batch = BatchedInputDataset(*[ex.to(device) for ex in batch])

            # Forward pass.
            shape = batch.input_features.shape
            features_flat = batch.input_features.view(-1, shape[-1])
            logits = selector(features_flat.float()).view(*shape[:-1])
            weights = torch.sigmoid(logits)

            all_outputs.append(batch.confidences.cpu())
            all_targets.append(batch.labels.cpu())
            all_weights.append(weights.cpu())

    # Aggregate.
    combine = torch.cat if is_batched else torch.stack
    all_outputs = combine(all_outputs, dim=0)
    all_targets = combine(all_targets, dim=0)
    all_weights = combine(all_weights, dim=0)

    # Compute metrics.
    all_metrics = collections.defaultdict(list)
    if coverage <= 0:
        eval_fn = metrics.compute_metric_auc
    else:
        eval_fn = functools.partial(
            metrics.compute_metric_at_coverage, coverage)
    for idx in range(len(all_outputs)):
        for metric, metric_fn in METRICS:
            all_metrics[metric].append(eval_fn(
                outputs=all_outputs[idx],
                targets=all_targets[idx],
                weights=all_weights[idx],
                metric_fn=metric_fn))

    # Average results per metric across datasets.
    all_metrics = {k: metrics.reduce_mean(v) for k, v in all_metrics.items()}


def main(args):
    torch.manual_seed(1)
    np.random.seed(1)
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    args.cuda = torch.cuda.is_available() and not args.use_cpu
    print('Using CUDA' if args.cuda else 'Using CPU')
    device = torch.device('cuda:0' if args.cuda else 'cpu')

    results = collections.defaultdict(list)
    for filename in args.datasets:
        basename = os.path.basename(filename)
        print(f'Evaluating {basename}')
        dataset = torch.utils.data.TensorDataset(*torch.load(filename))

        # Evaluate model and data bootstrap samples.
        with tqdm.tqdm(total=len(args.model_files) * args.bootstraps):
            for model_file in args.model_files:
                checkpoint = torch.load(model_file)
                input_dim, hidden_dim, num_layers = checkpoint["model"]
                selector = models.SelectiveNet(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers)
                selector.load_state_dict(checkpoint["state_dict"])
                selector = selector.to(device)
                for bootstrap in args.num_bootstraps:
                    indices = np.random.randint(len(dataset), size=len(dataset))
                    subset = torch.utils.data.Subset(dataset, indices)
                    loader = torch.utils.data.DataLoader(
                        dataset=subset,
                        batch_size=args.batch_size,
                        shuffle=False,
                        pin_memory=args.cuda)
                    results[basename].append(evaluate_model(selector, loader))

    torch.save(results, args.output_file)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
