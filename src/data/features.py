"""Processing meta features for use in the selector."""

import functools
import multiprocessing
import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import KernelDensity
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

import torch
import tqdm

from src.data import InputDataset, BatchedInputDataset

MODELS = None
OUTPUT_PROBS = None
INPUT_FEATURES = None

# Limit on training examples for sklearn models, for computational efficiency.
MAX_TRAIN_EXAMPLES = 20000

# Number of examples to process at a time.
BATCH_SIZE = 1024


# ------------------------------------------------------------------------------
#
# Helper functions to compute meta scores.
#
# ------------------------------------------------------------------------------


class FeatureScaler:
    """Standard feature scaler to mean 0 and unit variance."""

    def __init__(self, binary_columns=True):
        self.binary_columns = binary_columns

    def fit(self, X):
        print('\nFitting scaling...')
        self.mu = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        if self.binary_columns:
            binary_columns = []
            for col in range(X.shape[1]):
                vals = np.unique(X[:, col])
                if len(vals) > 2:
                    continue
                binary_columns.append(col)
            print(f'Num binary columns = {len(binary_columns)}')
            self.mu[binary_columns] = 0
            self.std[binary_columns] = 1

        # Fix features with std 0 to be 1.
        self.std[self.std == 0] = 1

        return self

    def scale(self, X):
        return (X - self.mu) / self.std


class KNNDistance:
    """Compute average distance to k nearest neighbors in training set."""

    def __init__(self, k=8):
        """Initialize class with number of neighbors to average over."""
        self.k = k

    def fit(self, source_vecs):
        """Store training set vectors."""
        self.source_vecs = torch.from_numpy(source_vecs).unsqueeze(0)

    def score_samples(self, query_vecs):
        """Score input examples based on average distance to source vectors."""
        query_vecs = torch.from_numpy(query_vecs).unsqueeze(0)
        dists = torch.cdist(query_vecs, self.source_vecs).squeeze(0)
        knn_dists = torch.topk(dists, self.k, dim=1, largest=False).values
        return knn_dists.mean(dim=1).view(-1).numpy()


def entropy(probs):
    """Compute entropy of predicted distribution."""
    probs = np.clip(probs, a_min=1e-8, a_max=1)
    return -np.sum(probs * np.log(probs), axis=-1)


def get_outlier_models(train_input_features):
    """Fit all outlier models used to derive scores as meta features."""
    scaler = FeatureScaler().fit(train_input_features)

    # For efficiency, we now subsample the train data.
    np.random.seed(1)
    indices = np.random.permutation(len(train_input_features))
    train_input_features = train_input_features[indices[:MAX_TRAIN_EXAMPLES]]
    train_input_features = scaler.scale(train_input_features)

    print('\nFitting KNN...')
    knn = KNNDistance()
    knn.fit(train_input_features)

    print('Fitting KDE...')
    kde = KernelDensity(kernel="gaussian", bandwidth=1.0, atol=0.01, rtol=0.01)
    kde = kde.fit(train_input_features)

    print('Fitting OSVM...')
    osvm = OneClassSVM(max_iter=10000)
    osvm.fit(train_input_features)

    print('Fitting Isolation Forest...')
    isoforest = IsolationForest()
    isoforest.fit(train_input_features)

    print('Fitting Local Outlier Factor...')
    lof = LocalOutlierFactor(novelty=True)
    lof.fit(train_input_features)

    return scaler, knn, kde, osvm, isoforest, lof


# ------------------------------------------------------------------------------
#
# Helper functions for processing dataset examples and computing meta scores.
#
# ------------------------------------------------------------------------------


def init(models, output_probs, input_features):
    """Initialize global shared models and dataset."""
    global MODELS, OUTPUT_PROBS, INPUT_FEATURES
    MODELS = models
    OUTPUT_PROBS = output_probs
    INPUT_FEATURES = input_features


def process_example(indices, skip_class_based_features=False):
    """Derive features for the i-th example.

    Args:
        indices: The indices of the examples to be processed (in DATASET).
        skip_class_based_features: Whether to skip full confidence/class info.

    Returns:
        Meta features vector.
    """
    global MODELS, OUTPUT_PROBS, INPUT_FEATURES
    scaler, knn, kde, osvm, isoforest, lof = MODELS
    probs, input_features = OUTPUT_PROBS[indices], INPUT_FEATURES[indices]

    # Gather meta features.
    meta_features = []

    # Class-based features.
    if not skip_class_based_features:
        meta_features.append(probs)
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(len(indices)), np.argmax(probs, axis=1)] = 1
        meta_features.append(one_hot)

    # Confidence-based features.
    meta_features.append(np.max(probs, axis=-1, keepdims=True))
    meta_features.append(entropy(probs).reshape(-1, 1))

    # Outlier/novelty-based features.
    features = scaler.scale(input_features)
    meta_features.append(knn.score_samples(features).reshape(-1, 1))
    meta_features.append(kde.score_samples(features).reshape(-1, 1))
    meta_features.append(osvm.score_samples(features).reshape(-1, 1))
    meta_features.append(isoforest.score_samples(features).reshape(-1, 1))
    meta_features.append(lof.score_samples(features).reshape(-1, 1))

    # Concatenate.
    meta_features = np.concatenate(meta_features, axis=-1)

    return indices, meta_features


def generate_dataset_features(
    models,
    dataset,
    skip_class_based_features=False,
    num_workers=1,
):
    """Compute meta features for all examples in a dataset split.

    Args:
        models: List of meta models (scaler, knn, kde, osvm, isoforest, lof).
        dataset: Either a InputDataset or BatchedInputDataset instance.
        skip_class_based_features: Whether to skip full confidence/class info.
        num_workers: Number of parallel processes to use for computation.

    Returns:
        [n x meta_dim] tensor of derived meta features.
    """
    input_features = dataset.input_features.numpy()
    output_probs = dataset.output_probs.numpy()

    # Flatten.
    leading_dims = input_features.shape[:-1]
    input_feature_dim = input_features.shape[-1]
    input_features = input_features.reshape(-1, input_feature_dim)
    output_probs = output_probs.reshape(-1, output_probs.shape[-1])

    # Initialize threads.
    if num_workers > 1:
        workers = multiprocessing.Pool(
            num_workers,
            initializer=init,
            initargs=(models, output_probs, input_features))
        map_fn = workers.imap_unordered
    else:
        init(models, output_probs, input_features)
        map_fn = map

    # Process example function with options.
    fn = functools.partial(
        process_example, skip_class_based_features=skip_class_based_features)

    meta_features = None
    num_examples = output_probs.shape[0]
    batches = np.array_split(
        np.arange(num_examples),
        np.arange(BATCH_SIZE, num_examples, BATCH_SIZE))
    with tqdm.tqdm(total=len(batches), desc='processing examples') as pbar:
        for idx, res in map_fn(fn, batches):
            if meta_features is None:
                meta_features = np.zeros((num_examples, res.shape[-1]))
            meta_features[idx] = res
            pbar.update()

    if num_workers > 0:
        workers.close()

    meta_features = meta_features.reshape(*leading_dims, -1)

    return torch.from_numpy(meta_features)


# ------------------------------------------------------------------------------
#
# Process all dataset examples (and all splits).
#
# ------------------------------------------------------------------------------


def process_dataset_splits(
    train_dataset,
    cal_dataset,
    val_dataset,
    test_datasets,
    skip_class_based_features=False,
    num_workers=1,
):
    """Process all given dataset splits to derive meta input features.

    Args:
        train_dataset: A InputDataset for the training set.
        cal_dataset: A BatchedInputDataset for the calibration set.
        val_dataset: A BatchedInputDataset for the validation set.
        test_datasets: A list of InputDataset for all testing sets.
        skip_class_based_features: Whether to skip full confidence/class info.
        num_workers: Number of parallel processes to use for computation.

    Returns:
        cal_meta_dataset: an instance of BatchedInputDataset.
        val_meta_dataset: an instance of BatchedInputDataset.
        test_meta_datasets: a list of InputDatasets.
    """
    # Initialize models using training data.
    print('Getting derived models...')
    models = get_outlier_models(train_dataset.input_features.numpy())

    # --------------------------------------------------------------------------
    #
    # Compute calibration set first.
    #
    # --------------------------------------------------------------------------
    print('\nDeriving calibration features...')
    cal_features = generate_dataset_features(
        models=models,
        dataset=cal_dataset,
        skip_class_based_features=skip_class_based_features,
        num_workers=num_workers)

    cal_features = cal_features.numpy()
    feature_dim = cal_features.shape[-1]
    meta_scaler = FeatureScaler().fit(cal_features.reshape(-1, feature_dim))

    cal_features = torch.from_numpy(meta_scaler.scale(cal_features))
    cal_meta_dataset = BatchedInputDataset(
        input_features=cal_features,
        output_probs=cal_dataset.output_probs,
        confidences=cal_dataset.confidences,
        labels=cal_dataset.labels)

    # --------------------------------------------------------------------------
    #
    # Compute validation set next.
    #
    # --------------------------------------------------------------------------
    print('\nDeriving calibration features...')
    val_features = generate_dataset_features(
        models=models,
        dataset=val_dataset,
        skip_class_based_features=skip_class_based_features,
        num_workers=num_workers)

    val_features = torch.from_numpy(meta_scaler.scale(val_features.numpy()))
    val_meta_dataset = BatchedInputDataset(
        input_features=val_features,
        output_probs=val_dataset.output_probs,
        confidences=val_dataset.confidences,
        labels=val_dataset.labels)

    # --------------------------------------------------------------------------
    #
    # Compute all test datasets.
    #
    # --------------------------------------------------------------------------
    print(f'\nDeriving test features for {len(test_datasets)} datasets...')
    test_meta_datasets = []
    for dataset in test_datasets:
        features = generate_dataset_features(
            models=models,
            dataset=dataset,
            skip_class_based_features=skip_class_based_features,
            num_workers=num_workers)
        features = torch.from_numpy(meta_scaler.scale(features.numpy()))
        test_meta_datasets.append(InputDataset(
            input_features=features,
            output_probs=dataset.output_probs,
            confidences=dataset.confidences,
            labels=dataset.labels))

    return cal_meta_dataset, val_meta_dataset, test_meta_datasets
