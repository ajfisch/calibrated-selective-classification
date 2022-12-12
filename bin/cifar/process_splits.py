"""Process datasets to be in the desired format.

For each dataset we compute a tuple of:
    (1) Input features (x, or a featurization phi(x) using the given network).
    (2) Output probabilities p(y | x) for every y \in [K], assuming K classes.
    (3) Model confidence f(X), here defined as max p(y | x).
    (3) Target label Y, here defined as 1{y = argmax p(y | x)}.

See src.data.InputDataset.
"""

import argparse
import functools
import collections
import os
import numpy as np

from third_party.ResNeXt_DenseNet.models.densenet import densenet
from third_party.ResNeXt_DenseNet.models.resnext import resnext29
from third_party.WideResNet_pytorch.wideresnet import wideresnet

import torch
from src.data import image_datasets

parser = argparse.ArgumentParser()

parser.add_argument(
    '--model', type=str, default='wrn',
    choices=['wrn', 'densenet', 'resnext'],
    help='Model architecture type.')

parser.add_argument(
    '--model-file', type=str,
    default='data/models/cifar/base/standard/model_best.pth.tar',
    help='Filename for saved model.')

parser.add_argument(
    '--data-dir', type=str, default='data/processed/cifar',
    help='Directory where dataset splits (e.g., test_images.npy) are saved.')

parser.add_argument(
    '--output-dir', type=str, default='data/processed/cifar',
    help='Directory where processed outputs will be saved.')

parser.add_argument(
    '--num-cal-batches', type=int, default=50000,
    help='Number of perturbed batches to sample for calibration.')

parser.add_argument(
    '--num-val-batches', type=int, default=1000,
    help='Number of perturbed batches to sample for validation.')

parser.add_argument(
    '-batch-size', type=int, default=1024,
    help='Number of examples per batch, also per "perturbed dataset".')

parser.add_argument(
    '--temperature-scale', action='store_true',
    help='Perform temperature scaling over raw f(X) network outputs.')

parser.add_argument(
    '--num-temperature-scaling-batches', type=int, default=100,
    help='Number of batches to use for temperature scaling.')

parser.add_argument(
    '--use-augmix-with-temperature-scaling', action='store_true',
    help='Perform temperature scaling using AugMix perturbed data.')

parser.add_argument(
    '--mixture-width', type=int, default=3,
    help='Number of augmentation chains to sample and mix.')

parser.add_argument(
    '--mixture-depth', type=int, default=-1,
    help='Number of augmentations per chain. Unif[1, 3] if set to -1.')

parser.add_argument(
    '--aug-severity', type=int, default=5,
    help='Augmentation level, higher is stronger.')

parser.add_argument(
    '--independent_perturbations', action='store_true',
    help='Do not share perturbations across "datasets"/sample per example.')

parser.add_argument(
    '--use-cpu', action='store_true',
    help='Run on CPU instead of GPU.')

parser.add_argument(
    '--batch-size', type=int, default=512,
    help='Batch size to use over GPU when deriving features/output probs.')

parser.add_argument(
    '--num-workers', type=int, default=16,
    help='Number of processes to use during data loading.')


def main(args):
    torch.manual_seed(1)
    np.random.seed(1)

    print(f'Will save results to {args.output_dir}.')
    os.makedirs(args.output_dir, exist_ok=True)

    # --------------------------------------------------------------------------
    #
    # Load datasets and base model.
    #
    # --------------------------------------------------------------------------
    print('Loading datasets...')
    splits = {
        'train': image_datasets.CIFAR10Dataset(
            os.path.join(args.data_dir, 'train_images.npy'),
            os.path.join(args.data_dir, 'train_labels.npy')),
        'cal': image_datasets.CIFAR10Dataset(
            os.path.join(args.data_dir, 'cal_images.npy'),
            os.path.join(args.data_dir, 'cal_labels.npy')),
        'val': image_datasets.CIFAR10Dataset(
            os.path.join(args.data_dir, 'val_images.npy'),
            os.path.join(args.data_dir, 'val_labels.npy')),
        'test_clean': image_datasets.CIFAR10Dataset(
            os.path.join(args.data_dir, 'test_images.npy'),
            os.path.join(args.data_dir, 'test_labels.npy')),
    }
    for corruption in image_datasets.CIFAR_CORRUPTIONS:
        splits[f'test_{corruption}'] = image_datasets.CIFAR10Dataset(
            os.path.join(args.data_dir, 'CIFAR-10-C', f'{corruption}.npy'),
            os.path.join(args.data_dir, 'CIFAR-10-C', 'labels.npy'))

    print('Initializing model...')
    if args.model == 'densenet':
        net = densenet(num_classes=10)
    elif args.model == 'wrn':
        net = wideresnet(num_classes=10)
    elif args.model == 'resnext':
        net = resnext29(num_classes=10)
    device = torch.device('cuda:0' if args.cuda else 'cpu')
    net = net.to(device)
    net.device = device

    # Load from checkpoint.
    checkpoint = torch.load(args.model_file)
    state_dict = collections.OrderedDict()
    for k, v in checkpoint['state_dict'].items():
        if k.startswith('module'):
            k = k[len('module.'):]
        state_dict[k] = v
    net.load_state_dict(state_dict)
    print('Base model weights loaded from:', args.model_file)

    # --------------------------------------------------------------------------
    #
    # Optionally apply temperature scaling.
    #
    # --------------------------------------------------------------------------
    if args.temperature_scale:
        print('\nTemperature scaling...')
        # Either use the augmix dataloader or the clean one.
        if args.use_augmix_with_temperature_scaling:
            dataloader_fn = functools.partial(
                image_datasets.get_augmix_image_dataloader,
                mixture_width=args.mixture_width,
                mixture_depth=args.mixture_depth,
                severity=args.aug_severity,
                independent_perturbations=True)
        else:
            dataloader_fn = image_datasets.get_clean_image_dataloader

        # Note: Temp. scaling reuses the same calibration data used later.
        loader = dataloader_fn(
            dataset=splits['cal'],
            batch_size=args.batch_size,
            num_batches=args.num_temperature_scaling_batches,
            num_workers=args.num_workers,
            pin_memory=args.cuda)

        # Optimize the temperature for temp. scaling.
        temperature = image_datasets.compute_temperature(net, loader)
        print(f'\tTemperature = {temperature:2.4f}')
    else:
        temperature = 1

    # --------------------------------------------------------------------------
    #
    # Process a limited amount of training data for features/derived models.
    #
    # --------------------------------------------------------------------------
    print('\nProcessing training data.')
    loader = image_datasets.get_clean_image_dataloader(
        dataset=splits['train'],
        batch_size=args.batch_size,
        num_batches=-1,  # Take whole dataset.
        num_workers=args.num_workers,
        pin_memory=args.cuda)
    ds = image_datasets.convert_image_dataset(
        net, loader, temperature, keep_batches=False)
    torch.save(ds, os.path.join(args.output_dir, f'train.pt'))

    # --------------------------------------------------------------------------
    #
    # Process perturbed datasets (calibration + validation data).
    #
    # --------------------------------------------------------------------------
    for split in ['cal', 'val']:
        print(f'\nProcessing pertubed {split} dataset.')
        loader = image_datasets.get_augmix_image_dataloader(
            dataset=splits[split],
            batch_size=args.batch_size,
            num_batches=(args.num_cal_batches if split == 'cal'
                         else args.num_val_batches),
            mixture_width=args.mixture_width,
            mixture_depth=args.mixture_depth,
            severity=args.aug_severity,
            independent_perturbations=False,
            num_workers=args.num_workers,
            pin_memory=args.cuda)
        ds = image_datasets.convert_image_dataset(
            net, loader, temperature, keep_batches=True)
        torch.save(ds, os.path.join(args.output_dir, f'p{split}.pt'))

    # --------------------------------------------------------------------------
    #
    # Process all test datasets.
    #
    # --------------------------------------------------------------------------
    print('\nProcessing test datasets.')
    for split in filter(lambda k: k.startswith('test'), splits.keys()):
        loader = image_datasets.get_clean_image_dataloader(
            dataset=splits[split],
            batch_size=args.batch_size,
            num_batches=-1,  # Take whole dataset.
            num_workers=args.num_workers,
            pin_memory=args.cuda)
        ds = image_datasets.convert_image_dataset(
            net, loader, temperature, keep_batches=False)
        torch.save(ds, os.path.join(args.output_dir, f'{split}.pt'))


if __name__ == '__main__':
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available() and not args.use_cpu
    main(args)
