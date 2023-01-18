"""Print the initial metrics for input datasets (without selection)."""

import argparse
import collections
import functools
import os
import torch
import tqdm
from src import metrics

parser = argparse.ArgumentParser()
parser.add_argument(
    '--datasets', type=str, nargs='+',
    help='Paths to dataset to evaluate.')

parser.add_argument(
    '--is-binary', action='store_true',
    help='If true, label Y is treated as binary vs. "is correct" reduction.')

# Map of name to function for metrics we compute.
METRICS = collections.OrderedDict((
    # The l_2 calibration error.
    ('ce_2', functools.partial(metrics.compute_ece, pnorm=2)),
    # The l_inf calibration error.
    ('ce_inf', functools.partial(metrics.compute_ece, pnorm=float('inf'))),
    # The Brier score.
    ('brier', metrics.compute_brier_score),
    # The accuracy.
    ('acc', metrics.compute_accuracy),
))


def main(args):
    # Specify if is_binary in accuracy computation.
    if args.is_binary:
        METRICS['acc'] = functools.partial(
            metrics.compute_accuracy, is_binary=True)

    # Keep track of all results + the average.
    dataset_to_results = {'avg': {'full': collections.defaultdict(list)}}
    for dataset in tqdm.tqdm(args.datasets, 'evaluating datasets'):
        name = os.path.basename(dataset)
        dataset_to_results[name] = {'full': {}}

        dataset = torch.load(dataset)
        outputs = dataset.confidences
        targets = dataset.labels
        weights = torch.ones_like(outputs)
        if outputs.dim() == 1:
            outputs = outputs.unsqueeze(0)
            targets = targets.unsqueeze(0)
            weights = weights.unsqueeze(0)

        # Without selection = compute at coverage 1.
        all_metrics = collections.defaultdict(list)
        for idx in range(len(outputs)):
            for metric, metric_fn in METRICS.items():
                all_metrics[metric].append(
                    metrics.compute_metric_at_coverage(
                        outputs=outputs[idx],
                        targets=targets[idx],
                        weights=weights[idx],
                        coverage=1.0,
                        metric_fn=metric_fn))

        # If this is a BatchedInputDataset, then compute mean across batches.
        for k, v in all_metrics.items():
            v = metrics.reduce_mean_coverage(v)
            dataset_to_results[name]['full'][k] = v
            dataset_to_results['avg']['full'][k].append(v)

    for k, v in dataset_to_results['avg']['full'].items():
        v = metrics.reduce_mean_coverage(v)
        dataset_to_results['avg']['full'][k] = v

    print(metrics.format_metrics(dataset_to_results, METRICS.keys()))


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
