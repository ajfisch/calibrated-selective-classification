"""Process CIFAR-10 datasets to be in the desired format.

For each dataset we compute a tuple of:
    (1) Input features \phi(x), here defined as last layer network features.
    (2) Output probabilities p(y | x) for every y \in [K], here K = 10.
    (3) Model confidence f(X), here defined as max p(y | x).
    (4) Target label Y, here defined as 1{y = argmax p(y | x)}.

See src.data.InputDataset.
"""

import argparse
import collections
import os

import numpy as np
import torch

from third_party.ResNeXt_DenseNet.models.densenet import densenet
from third_party.ResNeXt_DenseNet.models.resnext import resnext29
from third_party.WideResNet_pytorch.wideresnet import wideresnet
import third_party.augmentations

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

image_datasets.add_argparse_arguments(parser)

third_party.augmentations.IMAGE_SIZE = 32


def main(args):
    torch.manual_seed(1)
    np.random.seed(1)

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

    # Load from checkpoint.
    checkpoint = torch.load(args.model_file)
    state_dict = collections.OrderedDict()
    for k, v in checkpoint['state_dict'].items():
        if k.startswith('module'):
            k = k[len('module.'):]
        state_dict[k] = v.cpu()
    net.load_state_dict(state_dict)
    print('Base model weights loaded from:', args.model_file)

    print(f'Will save results to {args.output_dir}.')
    os.makedirs(args.output_dir, exist_ok=True)
    image_datasets.process_image_splits(args, net, splits, args.output_dir)


if __name__ == '__main__':
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available() and not args.use_cpu
    main(args)
