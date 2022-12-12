"""Data loading functions for image datasets using AugMix perturbations."""

import functools

import PIL
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import tqdm

from third_party import augmentations
from src.data import InputDataset, BatchedInputDataset
from src import utils

CIFAR_CORRUPTIONS = [
    'brightness',
    'contrast',
    'defocus_blur',
    'elastic_transform',
    'fog',
    'frost',
    'gaussian_noise',
    'glass_blur',
    'impulse_noise',
    'jpeg_compression',
    'motion_blur',
    'pixelate',
    'shot_noise',
    'snow',
    'zoom_blur',
]

IMAGENET_CORRUPTIONS = [
    'contrast',
    'defocus_blur',
    'elastic_transform',
    'gaussian_noise',
    'glass_blur',
    'impulse_noise',
    'jpeg_compression',
    'motion_blur',
    'pixelate',
    'shot_noise',
    'zoom_blur'
]


# ------------------------------------------------------------------------------
#
# PyTorch dataset wrappers.
#
# ------------------------------------------------------------------------------


class CIFAR10Dataset(torch.utils.data.Dataset):
    """Dataset wrapper for CIFAR10."""

    def __init__(self, image_file, label_file):
        super(CIFAR10Dataset).__init__()
        self.data = np.load(image_file)
        self.targets = np.load(label_file)

    def __getitem__(self, i):
        image = PIL.Image.fromarray(self.data[i])
        target = self.targets[i]
        return image, target

    def __len__(self):
        return len(self.data)

    @staticmethod
    def transform(image):
        fn = transforms.Compose(
            [transforms.RandomHorizontalFlip(),
             transforms.RandomCrop(32, padding=4)])
        return fn(image)

    @staticmethod
    def normalize(image):
        fn = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize([0.5] * 3, [0.5] * 3)])
        return fn(image)

    @staticmethod
    def collate(batch):
        images, targets = zip(*batch)
        normalize = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize([0.5] * 3, [0.5] * 3)])
        images = torch.cat([normalize(image).unsqueeze(0) for image in images])
        targets = torch.LongTensor(np.array(targets))
        return images, targets


class ImageNetDataset(torch.utils.data.Dataset):
    """Dataset wrapper for ImageNet."""

    def __init__(self, dirname):
        super(ImageNetDataset).__init__()
        self.data = datasets.ImageFolder(dirname)

    def __getitem__(self, i):
        return self.data[i]

    def __len__(self):
        return len(self.data)

    @staticmethod
    def transform(image):
        fn = transforms.Compose(
            [transforms.RandomResizedCrop(224),
             transforms.RandomHorizontalFlip()])
        return fn(image)

    @staticmethod
    def normalize(image):
        fn = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225])])
        return fn(image)

    @staticmethod
    def collate(batch):
        images, targets = zip(*batch)
        normalize = transforms.Compose(
            [transforms.Resize(256),
             transforms.CenterCrop(224),
             transforms.ToTensor(),
             transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225])])
        images = torch.cat([normalize(image).unsqueeze(0) for image in images])
        targets = torch.LongTensor(np.array(targets))
        return images, targets


# ------------------------------------------------------------------------------
#
# Image augmentation.
#
# ------------------------------------------------------------------------------


def apply_augmix_to_images(
    batch,
    transform_fn,
    normalize_fn,
    mixture_width=3,
    mixture_depth=-1,
    severity=3,
    independent_perturbations=False,
):
    """Perform AugMix augmentations over a batch of examples.

    Args:
      batch: List of (PIL.Image, target) example pairs.
      transform_fn: Base image transform function (dataset specific).
      normalize_fn: Base image tensor normalization function (dataset specific).
      mixture_width: Number of augmentation chains to sample and mix.
      mixture_depth: Number of augmentations per chain. If negative, selects a
          random number between 1 and 3.
      severity: Augmentation level, higher is stronger.
      independent_perturbations: If true, then different augmentations chains
          are sampled for each image, and not shared across a batch.

    Returns:
      mixed_images: A batch of image tensors with augmentations applies.
      targets: A batch of prediction targets.
    """
    images, targets = zip(*batch)

    # Use augmentations list that does not overlap with ImageNet-C.
    aug_list = augmentations.augmentations

    def sample_op_chain():
        ws = np.float32(np.random.dirichlet([1] * mixture_width))
        m = np.float32(np.random.beta(1, 1))
        op_chains = []
        for i in range(mixture_width):
            if mixture_depth > 0:
                depth = mixture_depth
            else:
                depth = np.random.randint(1, 4)
            op_chains.append([np.random.choice(aug_list) for _ in range(depth)])
        return (ws, m, op_chains)

    if not independent_perturbations:
        fixed_perturbation = sample_op_chain()

    # Apply perturbations to images.
    mixed_images = []
    for image in images:
        image = transform_fn(image)

        # Either take a random perturbation, or keep the fixed one.
        if not independent_perturbations:
            ws, m, op_chains = fixed_perturbation
        else:
            ws, m, op_chains = sample_op_chain()

        # If severity is 0, then shortcut and keep original image.
        if severity == 0:
            mixed_image = image

        # Otherwise, apply perturbation.
        else:
            mixed_image = (1 - m) * normalize_fn(image)
            for i in range(mixture_width):
                image_aug = image.copy()
                for op in op_chains[i]:
                    image_aug = op(image_aug, severity)
                mixed_image += m * ws[i] * normalize_fn(image_aug)

        # Add to batch.
        mixed_images.append(mixed_image.unsqueeze(0))

    # Batch tensors.
    mixed_images = torch.cat(mixed_images)
    targets = torch.LongTensor(targets)

    return mixed_images, targets


# ------------------------------------------------------------------------------
#
# Data loaders (with and without augmentation).
#
# ------------------------------------------------------------------------------


def get_augmix_image_dataloader(
    dataset,
    batch_size,
    num_batches,
    mixture_width=3,
    mixture_depth=-1,
    severity=3,
    independent_perturbations=False,
    **kwargs,
):
    """
    Helper to construct a dataloader with augmix perturbations applied.

    Examples are randomly sampled with replacement.

    Args:
        dataset: A PyTorch dataset of (image, target) pairs.
        batch_size: Number of examples per batch.
        num_batches: Total number of batches to sample.
        mixture_width: Number of augmentation chains to sample and mix.
        mixture_depth: Number of augmentations per chain. If negative, then a
            random depth between 1 and 3 (inclusive) is selected.
        severity: Augmentation level (higher is stronger).
        independent_perturbations: If true, then different augmentations chains
            are sampled for each image, and not shared across a batch.
        kwargs: DataLoader keyword args.

    Returns:
        A DataLoader object.
    """
    # Construct perturbation function.
    perturb_fn = functools.partial(
        apply_augmix_to_images,
        transform_fn=dataset.transform,
        normalize_fn=dataset.normalize,
        mixture_width=mixture_width,
        mixture_depth=mixture_depth,
        severity=severity,
        independent_perturbations=independent_perturbations)

    # Sample random batches from base dataset.
    sampler = torch.utils.data.RandomSampler(
        dataset,
        replacement=True,
        num_samples=num_batches * batch_size)

    # Return data loader.
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=perturb_fn,
        sampler=sampler,
        **kwargs)

    return loader


def get_clean_image_dataloader(
    dataset,
    batch_size,
    num_batches=-1,
    **kwargs,
):
    """
    Helper to construct a dataloader *without* augmix perturbations applied.

    Examples are sampled without replacement, unless num_batches is given.

    Args:
        dataset: A PyTorch dataset of (image, target) pairs.
        batch_size: Number of examples per batch.
        num_batches: Total number of batches to sample. If this is not None/-1,
            then random sampling with replacement of num_batches * batch_size
            examples is done. Otherwise, the dataset is directly batched.
        kwargs: DataLoader keyword args.

    Returns:
        A DataLoader object.
    """
    num_batches = num_batches if num_batches is not None else -1
    if num_batches > 0:
        sampler = torch.utils.data.RandomSampler(
            dataset,
            replacement=True,
            num_samples=num_batches * batch_size)

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            collate_fn=dataset.collate,
            sampler=sampler,
            **kwargs)

    # Otherwise directly batch dataset.
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=dataset.collate,
        **kwargs)

    return loader


# ------------------------------------------------------------------------------
#
# Helpers for processing inputs and computing temperature scaling.
#
# ------------------------------------------------------------------------------


def compute_temperature(net, loader):
    """Do temperature scaling on an image dataset, given a network.

    Args:
        net: nn.Module that implements 'get_features' and 'get_logits'.
        loader: DataLoader yielding batches of images and targets.

    Returns:
        Optimized temperature.
    """
    net.eval()
    device = net.device

    all_logits = []
    all_targets = []
    with torch.no_grad():
        for images, targets in tqdm.tqdm(loader, desc='computing logits'):
            features = net.get_features(images.to(device))
            logits = net.get_logits(features)
            all_logits.append(logits.cpu())
            all_targets.append(targets)
    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    return utils.temperature_scale(all_logits, all_targets)


def convert_image_dataset(net, loader, temperature=1, keep_batches=True):
    """Convert a dataset of images into an InputDataset.

    Args:
        net: nn.Module that implements 'get_features' and 'get_logits'.
        loader: DataLoader yielding batches of images and targets.
        temperature: Value to use for temperature scaling.
        keep_batches: Stack rather than concatenate all input batches.

    Returns:
        An instance of InputDataset or BatchedInputDataset.
    """
    net.eval()
    device = net.device

    all_confidences = []
    all_features = []
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for images, targets in tqdm.tqdm(loader, desc='processing dataset'):
            images, targets = images.to(device), targets.to(device)
            features = net.get_features(images)
            probs = F.softmax(net.get_logits(features) / temperature, dim=1)
            confidences = torch.max(probs, dim=1)[0]
            labels = targets.eq(torch.argmax(probs, dim=1)).float()

            all_confidences.append(confidences.cpu())
            all_features.append(features.cpu())
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())

    combine_fn = torch.stack if keep_batches else torch.cat
    all_confidences = combine_fn(all_confidences, dim=0)
    all_features = combine_fn(all_features, dim=0)
    all_probs = combine_fn(all_probs, dim=0)
    all_labels = combine_fn(all_labels)

    cls = InputDataset if not keep_batches else BatchedInputDataset
    return cls(
        input_features=all_features,
        output_probs=all_probs,
        confidences=all_confidences,
        labels=all_labels)
