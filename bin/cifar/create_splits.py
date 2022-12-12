"""Simple script to split CIFAR10 to train/cal/val + test (fixed)."""

import argparse
import os
import numpy as np
import torchvision.datasets as datasets

parser = argparse.ArgumentParser()

parser.add_argument(
    '--data-dir', type=str, default='./data/datasets/CIFAR-10',
    help='Path to CIFAR-10 dataset (or where it should be downloaded).')

parser.add_argument(
    '--train-p', type=float, default=0.80,
    help='Percentage of training data to reserve for a proper train set.')

parser.add_argument(
    '--cal-p', type=float, default=0.10,
    help='Percentage of training data to reserve for a calibration set.')

parser.add_argument(
    '--val-p', type=float, default=0.10,
    help='Percentage of training data to reserve for a validationset.')

parser.add_argument(
    '--output-dir', type=str, default='./data/processed/cifar',
    help='Path where new splits will be saved.')

args = parser.parse_args()

np.random.seed(42)

# Load standard training split.
train_dataset = datasets.CIFAR10(args.data_dir, train=True, download=True)
train_images = train_dataset.data
train_labels = np.array(train_dataset.targets)

num_train = int(len(train_dataset) * args.train_p)
num_cal = int(len(train_dataset) * args.cal_p)
num_val = len(train_dataset) - num_train - num_cal

# Load standard test split.
test_dataset = datasets.CIFAR10(args.data_dir, train=False, download=True)
test_images = test_dataset.data
test_labels = np.array(test_dataset.targets)
num_test = len(test_dataset)

print(f'Num training = {num_train}')
print(f'Num calibration = {num_cal}')
print(f'Num validation = {num_val}')
print(f'Num testing = {num_test}')
print(f'Saving to {args.output_dir}...')

os.makedirs(args.output_dir, exist_ok=True)

# Train.
indices = np.random.permutation(len(train_dataset))
np.save(os.path.join(args.output_dir, 'train_images.npy'),
        train_images[indices[:num_train]])
np.save(os.path.join(args.output_dir, 'train_labels.npy'),
        train_labels[indices[:num_train]])
indices = indices[num_train:]

# Calibration.
np.save(os.path.join(args.output_dir, 'cal_images.npy'),
        train_images[indices[:num_cal]])
np.save(os.path.join(args.output_dir, 'cal_labels.npy'),
        train_labels[indices[:num_cal]])
indices = indices[num_cal:]

# Validation.
np.save(os.path.join(args.output_dir, 'val_images.npy'), train_images[indices])
np.save(os.path.join(args.output_dir, 'val_labels.npy'), train_labels[indices])

# Test.
np.save(os.path.join(args.output_dir, 'test_images.npy'), test_images)
np.save(os.path.join(args.output_dir, 'test_labels.npy'), test_labels)
