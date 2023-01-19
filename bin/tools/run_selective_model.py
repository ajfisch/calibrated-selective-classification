"""Generate selector predictions."""

import argparse
import os
import tempfile
import tqdm

import numpy as np
import torch

from src import models
from src import utils
from src.data import InputDataset

parser = argparse.ArgumentParser()

parser.add_argument(
    '--model-file', type=str,
    help='Path to model file.')

parser.add_argument(
    '--input-dataset', type=str,
    help='Path to dataset to evaluate.')

parser.add_argument(
    '--calibration-dataset', type=str, default=None,
    help='Path to dataset to use as calibration.')

parser.add_argument(
    '--output-file', type=str, default=None,
    help='Path to file where results will be saved.')

parser.add_argument(
    '--threshold', type=float,
    help='Pre-computed threshold to use to make selections.')

parser.add_argument(
    '--coverage', type=float, default=-1,
    help='Target coverage level.')

parser.add_argument(
    '--batch-size', type=int, default=128,
    help='Batch size for evaluating datasets.')

parser.add_argument(
    '--use-cpu', action='store_true',
    help='Force to run only on CPU.')


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

    print('Loading model...')
    checkpoint = torch.load(args.model_file)
    input_dim, hidden_dim, num_layers = checkpoint["model"]
    selector = models.SelectiveNet(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers)
    selector.load_state_dict(checkpoint["state_dict"])
    selector = selector.to(device)

    if args.threshold is None and args.coverage > 0:
        print('Loading calibration dataset...')
        dataset = torch.utils.data.TensorDataset(
            *torch.load(args.calibration_dataset or args.input_dataset))
        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
            shuffle=False)

        all_weights = []
        with torch.no_grad():
            for batch in tqdm.tqdm(loader, desc='generating weights'):
                batch = InputDataset(*[ex.to(device) for ex in batch])
                weights = torch.sigmoid(selector(batch.input_features))
                all_weights.append(weights.cpu().view(-1))
        all_weights = torch.cat(all_weights, dim=0)
        args.threshold = utils.calibrate(all_weights, args.coverage)
        print(f'Threshold = {args.threshold:.5f}')

    print('Loading target dataset...')
    dataset = torch.utils.data.TensorDataset(
        *torch.load(args.input_dataset))
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False)

    all_weights = []
    with torch.no_grad():
        for batch in tqdm.tqdm(loader, desc='generating weights'):
            batch = InputDataset(*[ex.to(device) for ex in batch])
            weights = torch.sigmoid(selector(batch.input_features))
            all_weights.append(weights.cpu().view(-1))
    all_weights = torch.cat(all_weights, dim=0)

    if args.threshold is not None:
        all_weights = all_weights.ge(args.threshold).float()

    torch.save(all_weights, args.output_file)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
