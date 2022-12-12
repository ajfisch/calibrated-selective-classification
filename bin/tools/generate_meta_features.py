"""Generate meta features from an input dataset."""

import argparse
import os

import numpy as np
import torch

from src.data import features as features_lib

parser = argparse.ArgumentParser()

parser.add_argument(
    '--train-dataset', type=str,
    help='Path to InputDataset storing training data.')

parser.add_argument(
    '--cal-dataset', type=str,
    help='Path to BatchedInputDataset storing perturbed calibration data.')

parser.add_argument(
    '--val-dataset', type=str,
    help='Path to BatchedInputDataset storing perturbed validation data.')

parser.add_argument(
    '--test-datasets', type=str, nargs='+', default=[],
    help='Paths to InputDataset storing testing data.')

parser.add_argument(
    '--skip-class-based-features', action='store_true',
    help='If true, do not add full confidence/class info as meta features.')

parser.add_argument(
    '--output-dir', type=str, default=None,
    help='Output dir. Saved in same dirs as input paths if left unspecified.')

parser.add_argument(
    '--num-workers', type=int, default=32,
    help='Number of parallel processes for processing.')


def save(original_path, output_dir, dataset):
    """Helper to save a dataset to disk."""
    dirname = os.path.dirname(original_path)
    basename, ext = os.path.splitext(os.path.basename(original_path))
    output_dir = output_dir if output_dir is not None else dirname
    torch.save(dataset, os.path.join(output_dir, f'{basename}.meta{ext}'))


def main(args):
    torch.manual_seed(1)
    np.random.seed(1)

    if args.output_dir is not None:
        print(f'Will save to {args.output_dir}.')
        os.makedirs(args.output_dir, exist_ok=True)

    # Load clean calibration data.
    print('Loading data...')
    train_dataset = torch.load(args.train_dataset)
    cal_dataset = torch.load(args.cal_dataset)
    val_dataset = torch.load(args.val_dataset)
    test_datasets = [torch.load(dataset) for dataset in args.test_datasets]

    # Get meta features.
    print('Generating meta features...')
    meta_datasets = features_lib.process_dataset_splits(
        train_dataset=train_dataset,
        cal_dataset=cal_dataset,
        val_dataset=val_dataset,
        test_datasets=test_datasets,
        skip_class_based_features=args.skip_class_based_features,
        num_workers=args.num_workers)

    # Save to disk.
    print('Saving new datasets...')
    cal_dataset, val_dataset, test_datasets = meta_datasets
    save(args.cal_dataset, args.output_dir, cal_dataset)
    save(args.val_dataset, args.output_dir, val_dataset)
    for orig, dataset in zip(args.test_datasets, test_datasets):
        save(orig, args.output_dir, dataset)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
