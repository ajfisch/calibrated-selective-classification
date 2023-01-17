"""Process ImageNet datasets to be in the desired format.

For each dataset we compute a tuple of:
    (1) Input features \phi(x), here defined as projected ResNet50 features.
    (2) Output probabilities p(y | x) for every y \in [K], where K = 1000.
    (3) Model confidence f(X), here defined as max p(y | x).
    (4) Target label Y, here defined as 1{y = argmax p(y | x)}.

See src.data.InputDataset.
"""

import argparse
import os

import numpy as np
from torchvision.models import resnet50, ResNet50_Weights
import torch
import torch.nn as nn

import third_party.augmentations
from src.data import image_datasets

parser = argparse.ArgumentParser()

parser.add_argument(
    '--data-dir', type=str, default='data/datasets/imagenet',
    help='Directory where dataset splits (e.g., test_images.npy) are saved.')

parser.add_argument(
    '--output-dir', type=str, default='data/processed/imagenet',
    help='Directory where processed outputs will be saved.')

image_datasets.add_argparse_arguments(parser)

third_party.augmentations.IMAGE_SIZE = 224


class ResNet50(nn.Module):
    """Pre-trained ResNet50 wrapper."""

    def __init__(self):
        super(ResNet50, self).__init__()
        self.net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

    def get_features(self, x):
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)

        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3(x)
        x = self.net.layer4(x)

        x = self.net.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def get_logits(self, x):
        return self.net.fc(x)


def main(args):
    torch.manual_seed(1)
    np.random.seed(1)

    print('Loading datasets...')
    # For train/cal/val we just use different random samples from train.
    splits = {
        'train': image_datasets.ImageNetDataset(
            os.path.join(args.data_dir, 'train')),
        'cal': image_datasets.ImageNetDataset(
            os.path.join(args.data_dir, 'train')),
        'val': image_datasets.ImageNetDataset(
            os.path.join(args.data_dir, 'train')),
        'test_clean': image_datasets.ImageNetDataset(
            os.path.join(args.data_dir, 'val')),
    }
    for corruption in image_datasets.IMAGENET_CORRUPTIONS:
        splits[f'test_{corruption}'] = image_datasets.ImageNetDataset(
            os.path.join(args.data_dir, 'imagenet-c-combined', f'{corruption}'))

    print('Initializing model...')
    net = ResNet50()

    print(f'Will save results to {args.output_dir}.')
    os.makedirs(args.output_dir, exist_ok=True)
    image_datasets.process_image_splits(args, net, splits, args.output_dir)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
