"""Calibrate threshold for selector."""

import argparse
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
    '--calibration-dataset', type=str,
    help='Path to dataset to use as calibration.')

parser.add_argument(
    '--coverage', type=float, default=0.8,
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

    print('Loading dataset...')
    dataset = torch.utils.data.TensorDataset(
        *torch.load(args.calibration_dataset))
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

    threshold = utils.calibrate(all_weights, args.coverage)
    print(f'Threshold = {threshold:.5f}')


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
