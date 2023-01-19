"""Evaluate selector and baselines."""

import argparse
import collections
import functools
import multiprocessing.dummy as d_mp
import os
import tempfile

import numpy as np
import torch
import torch.multiprocessing as t_mp
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
    '--output-file', type=str, default=None,
    help='Path to file where results will be saved.')

parser.add_argument(
    '--is-binary', action='store_true',
    help='If true, label Y is treated as binary vs. "is correct" reduction.')

parser.add_argument(
    '--bootstraps', type=int, default=5,
    help='Number of resamples from each test dataset (used for mean + std).')

parser.add_argument(
    '--coverage', type=float, default=-1,
    help='Target coverage level (or AUC if ≤ 0).')

parser.add_argument(
    '--batch-size', type=int, default=128,
    help='Batch size for evaluating datasets.')

parser.add_argument(
    '--use-cpu', action='store_true',
    help='Force to run only on CPU.')

parser.add_argument(
    '--num-workers', type=int, default=16,
    help='Number of processes used to compute final results.')

# Map of name to function for default metrics we compute.
METRICS = collections.OrderedDict((
    # The selective l_2 calibration error.
    ('ce_2', functools.partial(metrics.compute_ece, pnorm=2)),
    # The selective l_inf calibration error.
    ('ce_inf', functools.partial(metrics.compute_ece, pnorm=float('inf'))),
    # The selective Brier score.
    ('brier', metrics.compute_brier_score),
    # The selective accuracy.
    ('acc', metrics.compute_accuracy),
))

# Map of scoring name (including baseline) to sign and feature index.
# Selector requires running the model.
HIGHER = 1
LOWER = -1
METHODS = collections.OrderedDict((
    ('full', None),
    ('selector', None),
    ('confidence', (HIGHER, -7)),
    ('entropy', (LOWER, -6)),
    ('knn', (LOWER, -5)),
    ('kde', (HIGHER, -4)),
    ('osvm', (HIGHER, -3)),
    ('isoforest', (HIGHER, -2)),
    ('lof', (HIGHER, -1)),
))


def evaluate_model(selector, data_loader, coverage=-1, method='selector'):
    """Evaluate the selective calibration error AUC of the selector.

    Args:
        selector: SelectiveNet nn.Module implementing the selector g(X).
        data_loader: DataLoader for evaluation data, where each batch can be
            converted into either a BatchedInputDataset or a InputDataset.
        coverage: Desired coverage level (-1 = AUC).
        method: Option for deriving selective weights. One of METHODS.
        return_async: Returns a function that, when called, returns the results.

    Returns:
        Returns a dict with each metric computed as as either an AUCResult or a
        CoverageResult. If the input is an instance of a BatchedInputDataset
        (i.e., composed of multiple datasets), then the average is taken.
    """
    if method == 'selector':
        selector.eval()
        device = next(selector.parameters()).device
    else:
        device = torch.device('cpu')

    # Infer from the batches if the input is a BatchedInputDataset or not.
    is_batched = False

    # Compute all weights for the dataset.
    all_outputs, all_targets, all_weights = [], [], []

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

            # Get scores using the desired method.
            if method == 'selector':
                # Run the model.
                logits = selector(features_flat.float())
                weights = torch.sigmoid(logits)
            elif method == 'full':
                # Just take uniform (the same) weights.
                weights = torch.ones_like(features_flat[:, 0].float())
            else:
                # Select from features and adjust sign.
                sign, index = METHODS[method]
                weights = sign * features_flat[:, index].float()
            weights = weights.view(*shape[:-1])

            all_outputs.append(batch.confidences.cpu())
            all_targets.append(batch.labels.cpu())
            all_weights.append(weights.cpu())

    # Aggregate.
    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_weights = torch.cat(all_weights, dim=0)

    # Insert dummy dimension if input is a single dataset.
    if not is_batched:
        all_outputs = all_outputs.unsqueeze(0)
        all_targets = all_targets.unsqueeze(0)
        all_weights = all_weights.unsqueeze(0)

    # Determine calibration/scoring mechanism if evaluating at a specific
    # coverage, or over all coverages (as part of an AUC calculation).
    if coverage <= 0:
        eval_fn = metrics.compute_metric_auc
        reduce_fn = metrics.reduce_mean_auc
    else:
        eval_fn = functools.partial(
            metrics.compute_metric_at_coverage,
            coverage=coverage)
        reduce_fn = metrics.reduce_mean_coverage

    all_metrics = collections.defaultdict(list)
    for idx in range(len(all_outputs)):
        for metric, metric_fn in METRICS.items():
            all_metrics[metric].append(eval_fn(
                outputs=all_outputs[idx],
                targets=all_targets[idx],
                weights=all_weights[idx],
                metric_fn=metric_fn))

    # Average results per metric across datasets.
    # (Again, this is only a mean if the input is a BatchedInputDataset.)
    return {k: reduce_fn(v) for k, v in all_metrics.items()}


def main(args):
    torch.manual_seed(1)
    np.random.seed(1)
    if not args.output_file:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            args.output_file = tmp.name
    else:
        dirname = os.path.dirname(args.output_file)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
    print(f'Will save results to {args.output_file}')

    args.cuda = torch.cuda.is_available() and not args.use_cpu
    print('Using CUDA' if args.cuda else 'Using CPU')
    device = torch.device('cuda:0' if args.cuda else 'cpu')

    # Specify if is_binary in accuracy computation.
    if args.is_binary:
        METRICS['acc'] = functools.partial(
            metrics.compute_accuracy, is_binary=True)

    # Start a multiprocessing pool of workers.
    if args.cuda:
        t_mp.set_start_method('spawn', force=True)
    if args.num_workers > 0:
        pool = t_mp.Pool(args.num_workers)
    else:
        pool = d_mp.Pool(1)

    # Results for all files.
    # Will be a dict of dataset --> method --> metric --> result.
    dataset_to_results = collections.defaultdict(
        lambda: collections.defaultdict(lambda: collections.defaultdict(list)))

    # Initialize all evaluation jobs and add to queue.
    jobs = []
    total = len(args.model_files) * args.bootstraps * len(args.datasets)
    with tqdm.tqdm(total=total, desc='adding jobs to queue') as pbar:
        for filename in args.datasets:
            name = os.path.basename(filename)
            dataset = torch.utils.data.TensorDataset(*torch.load(filename))

            # Evaluate model and data bootstrap samples.
            for model_file in args.model_files:
                # Load model.
                checkpoint = torch.load(model_file)
                input_dim, hidden_dim, num_layers = checkpoint["model"]
                selector = models.SelectiveNet(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers)
                selector.load_state_dict(checkpoint["state_dict"])
                selector = selector.to(device)

                # Evaluate resamples of the underlying dataset.
                for bootstrap in range(args.bootstraps):
                    indices = np.random.randint(len(dataset), size=len(dataset))
                    subset = torch.utils.data.Subset(dataset, indices)
                    loader = torch.utils.data.DataLoader(
                        dataset=subset,
                        batch_size=args.batch_size,
                        shuffle=False)

                    # Evaluate each baseline method + selector.
                    for method in METHODS:
                        eval_fn = functools.partial(
                            evaluate_model,
                            selector=selector,
                            data_loader=loader,
                            coverage=args.coverage,
                            method=method)
                        jobs.append((name, method, pool.apply_async(eval_fn)))
                    pbar.update()

    # Wait for all results to compute.
    for name, method, results in tqdm.tqdm(jobs, desc='computing jobs'):
        results = results.get()
        for k, v in results.items():
            dataset_to_results[name][method][k].append(v)
            dataset_to_results['avg'][method][k].append(v)

    # Compute averages over samples (and all datasets).
    if args.coverage <= 0:
        reduce_fn = metrics.reduce_mean_auc
    else:
        reduce_fn = metrics.reduce_mean_coverage

    dataset_to_avg_result = {}
    for dataset, method_to_result in dataset_to_results.items():
        dataset_to_avg_result[dataset] = {}
        for method, results in method_to_result.items():
            dataset_to_avg_result[dataset][method] = {}
            for k, v in results.items():
                dataset_to_avg_result[dataset][method][k] = reduce_fn(v)

    print(metrics.format_metrics(dataset_to_avg_result, METRICS.keys()))
    torch.save(dataset_to_avg_result, args.output_file)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
