"""Train selector."""

import argparse
import collections
import tqdm
import os
import numpy as np
import torch

from src import metrics
from src import models

parser = argparse.ArgumentParser()

parser.add_argument(
    '--model-files', type=str, nargs='+',
    help='Paths to model files (if multiple, will be used to compute std).')

parser.add_argument(
    '--batch-size', type=int, default=128,
    help='Batch size for evaluating datasets.')

parser.add_argument(
    '--datasets', type=str, nargs='+',
    help='Paths to datasets to evaluate.')

parser.add_argument(
    '--bootstraps', type=int, default=5,
    help='Number of resamples from each test dataset.')

parser.add_argument(
    '--use-cpu', action='store_true',
    help='Force to run only on CPU.')

parser.add_argument(
    '--output-dir', type=str,
    help='Path to directory where results will be saved.')

args = parser.parse_args()
args.cuda = torch.cuda.is_available() and not args.use_cpu
device = torch.device("cuda:0" if args.cuda else "cpu")


def evaluate(selector, data_loader):
    """Test model."""
    selector.eval()
    all_confidences = []
    all_targets = []
    all_weights = collections.defaultdict(list)

    # Run over full dataset.
    with torch.no_grad():
        for inputs in data_loader:
            # Transfer to device.
            inputs = [i.to(device) for i in inputs]
            confidences = inputs[0]
            targets = inputs[1]
            features = inputs[-1]

            # Aggregate.
            all_confidences.append(confidences)
            all_targets.append(targets)

            # Baselines.
            for name, (sign, index) in BASELINES.items():
                all_weights[name].append(sign * features[:, index])

            # Selector.
            all_weights["selective"].append(selector(features))

    all_confidences = torch.cat(all_confidences, dim=0).unsqueeze(0)
    all_targets = torch.cat(all_targets, dim=0).unsqueeze(0)
    for name, weights in all_weights.items():
        all_weights[name] = torch.cat(weights, dim=0).unsqueeze(0)

    results = collections.OrderedDict()
    results["full"] = {
        "brier_auc": 100 * metrics.compute_brier(
            confidences=all_confidences[0],
            correctness=all_targets[0]),
        "ece_auc": 100 * metrics.compute_ece(
            confidences=all_confidences[0],
            correctness=all_targets[0],
            p=2),
        "mce_auc": 100 * metrics.compute_ece(
            confidences=all_confidences[0],
            correctness=all_targets[0],
            p="max"),
        "acc_auc": 100 * metrics.compute_accuracy(
            correctness=all_targets[0]),
    }
    for name, weights in all_weights.items():
        eces = []
        mces = []
        briers = []
        accs = []
        for coverage in np.arange(0.0, 1.01, 0.01):
            ece, mce, brier, acc = metrics.compute_marginal_metrics(
                confidences=all_confidences,
                correctness=all_targets,
                weights=weights,
                coverage=coverage)["selective"]
            briers.append((coverage, brier[0]))
            eces.append((coverage, ece[0]))
            mces.append((coverage, mce[0]))
            accs.append((coverage, acc[0]))
        ece_auc = metrics.compute_auc(eces)
        mce_auc = metrics.compute_auc(mces)
        brier_auc = metrics.compute_auc(briers)
        acc_auc = metrics.compute_auc(accs)
        results[name] = {
            "brier": briers,
            "brier_auc": brier_auc,
            "ece": eces,
            "ece_auc": ece_auc,
            "mce": mces,
            "mce_auc": mce_auc,
            "acc": accs,
            "acc_auc": acc_auc,
        }

    return results


def main():
    """Main script for training and evaluation."""
    torch.manual_seed(1)
    np.random.seed(1)
    os.makedirs(args.save, exist_ok=True)

    all_results = collections.defaultdict(list)
    total = len(args.model_files) * args.bootstraps * (len(CORRUPTIONS) + 1)
    with tqdm.tqdm(total=total) as pbar:
        # Do for each model and bootstrap.
        for model_file in args.model_files:
            checkpoint = torch.load(model_file)
            input_dim, hidden_dim, num_layers = checkpoint["model"]
            selector = models.SelectiveNet(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers)
            selector.load_state_dict(checkpoint["state_dict"])
            selector = selector.to(device)

            for name in ["clean"] + CORRUPTIONS:
                filename = f"test_{name}.pt"
                path = os.path.join(checkpoint["data"], filename)
                dataset = load_dataset(path)
                for bootstrap in range(args.bootstraps):
                    indices = np.random.randint(len(dataset), size=len(dataset))
                    subset = torch.utils.data.Subset(dataset, indices)
                    loader = torch.utils.data.DataLoader(
                        dataset=subset,
                        batch_size=args.batch_size,
                        shuffle=False,
                        pin_memory=args.cuda)
                    results = evaluate(selector, loader)
                    all_results[name].append(results)
                    pbar.update()
    torch.save(all_results, os.path.join(args.save, "all_results.pt"))


if __name__ == "__main__":
  main()
