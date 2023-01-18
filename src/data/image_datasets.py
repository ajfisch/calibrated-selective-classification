"""Data loading functions for image datasets using AugMix perturbations."""

import functools
import os

from sklearn.decomposition import TruncatedSVD
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

GPU_BATCH_SIZE = 256


def set_gpu_batch_size(batch_size):
    """Reset batch size constant."""
    global GPU_BATCH_SIZE
    GPU_BATCH_SIZE = batch_size


# ------------------------------------------------------------------------------
#
# PyTorch dataset wrappers.
#
# ------------------------------------------------------------------------------


class CIFAR10Dataset(torch.utils.data.Dataset):
    """Dataset wrapper for CIFAR10."""

    def __init__(self, image_file, label_file, transform=None):
        super(CIFAR10Dataset).__init__()
        self.data = np.load(image_file)
        self.targets = np.load(label_file)

        # Backwards compatibility for train_base_model.
        self.given_transform = transform

    def __getitem__(self, i):
        image = PIL.Image.fromarray(self.data[i])

        # Backwards compatibility for train_base_model.
        if self.given_transform is not None:
            image = self.given_transform(image)

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


def compute_subbatches(net, images, feature_projection=None):
    """Compute batch features and logits in increments.

    Args:
        net: nn.Module that implements 'get_features' and 'get_logits'.
        images: Batch of image tensors.
        feature_projection: Projection matrix to apply to features.

    Returns:
        Features and logits for the input images.
    """
    net.eval()
    device = next(net.parameters()).device
    batch_size = len(images)
    indices = list(range(batch_size))
    splits = [indices[i:i + GPU_BATCH_SIZE]
              for i in range(0, batch_size, GPU_BATCH_SIZE)]

    features, logits = [], []
    for split in splits:
        split_features = net.get_features(images[split].to(device))
        split_logits = net.get_logits(split_features)

        # Maybe project features...
        if feature_projection is not None:
            split_features = torch.mm(split_features, feature_projection)

        features.append(split_features.cpu())
        logits.append(split_logits.cpu())

    return torch.cat(features, dim=0), torch.cat(logits, dim=0)


def compute_temperature(net, loader):
    """Do temperature scaling on an image dataset, given a network.

    Args:
        net: nn.Module that implements 'get_features' and 'get_logits'.
        loader: DataLoader yielding batches of images and targets.

    Returns:
        Optimized temperature.
    """
    all_logits = []
    all_targets = []
    with torch.no_grad():
        for images, targets in tqdm.tqdm(loader, desc='computing logits'):
            features, logits = compute_subbatches(net, images)
            all_logits.append(logits)
            all_targets.append(targets)
    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    temperature = utils.temperature_scale(all_logits, all_targets)

    return temperature


def compute_svd(net, loader, n_components=128):
    """Compute a truncated SVD on an image dataset, given network features.

    Args:
        net: nn.Module that implements 'get_features' and 'get_logits'.
        loader: DataLoader yielding batches of images and targets.

    Returns:
        Projection matrix from TruncatedSVD.
    """
    device = next(net.parameters()).device
    all_features = []
    with torch.no_grad():
        for images, _ in tqdm.tqdm(loader, desc='computing features'):
            features, _ = compute_subbatches(net, images)
            all_features.append(features)
    all_features = torch.cat(all_features, dim=0).numpy()

    svd = TruncatedSVD(n_components=n_components, n_iter=50, random_state=42)
    svd.fit(all_features)

    return torch.from_numpy(svd.components_.T).to(device)


def convert_image_dataset(
    net,
    loader,
    temperature=1,
    keep_batches=True,
    svd=None,
):
    """Convert a dataset of images into an InputDataset.

    Args:
        net: nn.Module that implements 'get_features' and 'get_logits'.
        loader: DataLoader yielding batches of images and targets.
        temperature: Value to use for temperature scaling.
        keep_batches: Stack rather than concatenate all input batches.
        svd: Projection matrix of TruncatedSVD, as a torch tensor.

    Returns:
        An instance of InputDataset or BatchedInputDataset.
    """
    N, M = len(loader), loader.batch_size
    all_features = None
    all_probs = None
    all_confidences = torch.empty(N, M)
    all_labels = torch.empty(N, M)
    idx = count = 0
    with torch.no_grad():
        for images, targets in tqdm.tqdm(loader, desc='processing dataset'):
            features, logits = compute_subbatches(net, images, svd)
            probs = F.softmax(logits / temperature, dim=1)
            confidences = torch.max(probs, dim=1)[0]
            labels = targets.eq(torch.argmax(probs, dim=1)).float()

            if all_features is None:
                all_features = torch.empty(N, M, features.size(-1))
                all_probs = torch.empty(N, M, probs.size(-1))

            all_features[idx] = features
            all_confidences[idx] = confidences
            all_probs[idx] = probs
            all_labels[idx] = labels

            idx += 1
            count += len(images)

    if not keep_batches:
        all_features = all_features.view(N * M, -1)[:count]
        all_probs = all_probs.view(N * M, -1)[:count]
        all_confidences = all_confidences.view(N * M)[:count]
        all_labels = all_labels.view(N * M)[:count]

    cls = InputDataset if not keep_batches else BatchedInputDataset
    return cls(
        input_features=all_features,
        output_probs=all_probs,
        confidences=all_confidences,
        labels=all_labels)


# ------------------------------------------------------------------------------
#
# Helper to process all splits for a typical image classification dataset.
#
# ------------------------------------------------------------------------------


def add_argparse_arguments(parser):
    """Add arguments specific to this library."""
    parser.add_argument(
        '--num-clean-train-batches', type=int, default=1000,
        help='Number of train batches to sample for deriving features, etc.')

    parser.add_argument(
        '--num-cal-batches', type=int, default=50000,
        help='Number of perturbed batches to sample for calibration.')

    parser.add_argument(
        '--num-val-batches', type=int, default=1000,
        help='Number of perturbed batches to sample for validation.')

    parser.add_argument(
        '--batch-size', type=int, default=1024,
        help='Number of examples per batch, also per "perturbed dataset".')

    parser.add_argument(
        '--temperature-scale', action='store_true',
        help='Perform temperature scaling over raw f(X) network outputs.')

    parser.add_argument(
        '--num-temperature-scaling-batches', type=int, default=100,
        help='Number of batches to use for temperature scaling.')

    parser.add_argument(
        '--apply-svd', action='store_true',
        help='Perform truncated SVD to project features to lower dimension.')

    parser.add_argument(
        '--num-svd-batches', type=int, default=100,
        help='Number of batches to use for SVD.')

    parser.add_argument(
        '--num-svd-components', type=int, default=128,
        help='Dimensionality of SVD projection.')

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
        '--num-workers', type=int, default=20,
        help='Number of processes to use during data loading.')

    return parser


def process_image_splits(args, net, splits, output_dir):
    """Process all image splits given a splits dict and network.

    Args:
        args: Namespace containing args listed in "add_argparse_arguments".
        net: nn.Module that implements 'get_features' and 'get_logits'.
        splits: Dict with 'train', 'cal', 'val', and 'test_*' splits.
        output_dir: Where to save processed results.
    """
    args.cuda = torch.cuda.is_available() and not args.use_cpu
    print(f"\n[Using {'CUDA' if args.cuda else 'CPU'}]")
    device = torch.device('cuda:0' if args.cuda else 'cpu')
    net = net.to(device)

    # Optionally apply temperature scaling and SVD.
    if args.temperature_scale:
        print('\n[Temperature scaling]')
        # Either use the augmix dataloader or the clean one.
        if args.use_augmix_with_temperature_scaling:
            dataloader_fn = functools.partial(
                get_augmix_image_dataloader,
                mixture_width=args.mixture_width,
                mixture_depth=args.mixture_depth,
                severity=args.aug_severity,
                independent_perturbations=True)
        else:
            dataloader_fn = get_clean_image_dataloader

        # Note: Temp. scaling reuses the same calibration data used later.
        loader = dataloader_fn(
            dataset=splits['cal'],
            batch_size=args.batch_size,
            num_batches=args.num_temperature_scaling_batches,
            num_workers=args.num_workers,
            pin_memory=args.cuda)

        # Optimize the temperature for temp. scaling.
        temperature = compute_temperature(net, loader)
        print(f'Temperature = {temperature:2.4f}')
    else:
        temperature = 1

    if args.apply_svd:
        print('\n[Computing SVD]')

        # Note: SVD also reuses the same calibration data used later.
        loader = get_clean_image_dataloader(
            dataset=splits['cal'],
            batch_size=args.batch_size,
            num_batches=args.num_svd_batches,
            num_workers=args.num_workers,
            pin_memory=args.cuda)

        # Compute SVD.
        svd = compute_svd(net, loader, args.num_svd_components)
    else:
        svd = None

    # Process a limited amount of training data for features/derived models.
    print('\n[Processing training data]')
    loader = get_clean_image_dataloader(
        dataset=splits['train'],
        batch_size=args.batch_size,
        num_batches=args.num_clean_train_batches,
        num_workers=args.num_workers,
        pin_memory=args.cuda)
    ds = convert_image_dataset(
        net, loader, temperature, keep_batches=False, svd=svd)
    torch.save(ds, os.path.join(output_dir, f'train.pt'))
    del ds

    # Process perturbed datasets (calibration + validation data).
    for split in ['cal', 'val']:
        print(f'\n[Processing pertubed {split} dataset]')
        loader = get_augmix_image_dataloader(
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
        ds = convert_image_dataset(
            net, loader, temperature, keep_batches=True, svd=svd)
        torch.save(ds, os.path.join(output_dir, f'p{split}.pt'))
        del ds

    # Process all test datasets.
    print('\n[Processing test datasets]')
    for split in filter(lambda k: k.startswith('test'), splits.keys()):
        loader = get_clean_image_dataloader(
            dataset=splits[split],
            batch_size=args.batch_size,
            num_batches=-1,  # Take whole dataset.
            num_workers=args.num_workers,
            shuffle=False,
            pin_memory=args.cuda)
        ds = convert_image_dataset(
            net, loader, temperature, keep_batches=False, svd=svd)
        torch.save(ds, os.path.join(output_dir, f'{split}.pt'))
        del ds
