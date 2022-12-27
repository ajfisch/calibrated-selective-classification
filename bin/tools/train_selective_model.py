"""Train selector model."""

import argparse
import logging
import os
import shutil
import sys
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

from src import losses
from src import metrics
from src import utils
from src.data import BatchedInputDataset
from src.models import SelectiveNet

parser = argparse.ArgumentParser()

parser.add_argument(
    '--cal-dataset', type=str,
    help='Path to BatchedInputDataset storing perturbed calibration data.')

parser.add_argument(
    '--val-dataset', type=str,
    help='Path to BatchedInputDataset storing perturbed validation data.')

parser.add_argument(
    '--model-dir', type=str, default=None,
    help='Directory for saving model and logs.')

parser.add_argument(
    '--overwrite', action='store_true',
    help='Overwrite model_dir if it already exists.')

parser.add_argument(
    '--hidden-dim', type=int, default=64,
    help='Hidden dimension for selector MLP.')

parser.add_argument(
    '--num_layers', type=int, default=1,
    help='Number of layers for the selector MLP.')

parser.add_argument(
    '--dropout', type=float, default=0.0,
    help='Dropout level for the selector MLP.')

parser.add_argument(
    '--kappa', type=int, default=4,
    help='DRO kappa parameter. If -1, uses the whole batch (not kappa-worst).')

parser.add_argument(
    '--seed', type=int, default=42,
    help='Random seed to use for training.')

parser.add_argument(
    '--epochs', type=int, default=5,
    help='Number of training epochs.')

parser.add_argument(
    '--learning-rate', type=float, default=1e-3,
    help='Optimizer learning rate (for Adam).')

parser.add_argument(
    '--weight-decay', type=float, default=1e-5,
    help='L2 regularization strength (for Adam).')

parser.add_argument(
    '--train-batch-size', type=int, default=32,
    help='Batch size during training (batches of perturbed datasets).')

parser.add_argument(
    '--eval-batch-size', type=int, default=32,
    help='Batch size during evaluation (batches of perturbed datasets).')

parser.add_argument(
    '--p-norm', type=float, default=2,
    help='L_p norm parameter for use in S-MMCE and ECE calculations.')

parser.add_argument(
    '--smmce-weight', type=float, default=1,
    help='Weight for coverage regularizer (collapse of g).')

parser.add_argument(
    '--coverage-weight', type=float, default=1e-2,
    help='Weight for coverage regularizer (collapse of g).')

parser.add_argument(
    '--clip-grad-norm', type=float, default=10,
    help='Max grad norm for gradient clipping.')

parser.add_argument(
    '--print-freq', type=int, default=100,
    help='Print every n steps.')

parser.add_argument(
    '--use-cpu', action='store_true',
    help='Force to use CPU even if GPU is available.')

parser.add_argument(
    '--num-workers', type=int, default=8,
    help='Number of data loader background processes.')


def save_model(filename, selector):
    """Save model to file.

    Args:
        filename: Destination to save model to.
        selector: SelectiveNet nn.Module implementing the selector g(X).
    """
    checkpoint = {
        'state_dict': selector.cpu().state_dict(),
        'model': (selector.input_dim, selector.hidden_dim, selector.num_layers),
    }
    torch.save(checkpoint, filename)


def train_model(args, selector, data_loader, optimizer):
    """Train the selector model for one epoch.

    Args:
        args: Namespace object containing relevant training arguments.
        selector: SelectiveNet nn.Module implementing the selector g(X).
        data_loader: DataLoader for training data, where each batch can be
            converted into a BatchedInputDataset.
        optimizer: Torch optimizer for updating the selector.

    Returns:
        A reference to the selector.
    """
    selector.train()
    device = next(selector.parameters()).device

    loss_ema = smmce_ema = cov_ema = 0
    for i, batch in enumerate(data_loader):
        batch = BatchedInputDataset(*[ex.to(device) for ex in batch])

        # Forward pass.
        m, n = batch.labels.size()
        features_flat = batch.input_features.view(m * n, -1)
        logits = selector(features_flat.float()).view(m, n)
        weights = torch.sigmoid(logits)

        # Compute losses for each sub-dataset (perturbation) in the batch.
        smmce_loss = torch.zeros(m).to(device)
        cov_loss = torch.zeros(m).to(device)
        sbce_aucs = torch.zeros(m).to(device)
        for j in range(m):
            # Compute the S-MMCE_u loss over dataset j.
            smmce_loss[j] = losses.compute_smmce_loss(
                outputs=batch.confidences[j],
                targets=batch.labels[j],
                weights=weights[j],
                pnorm=args.p_norm)
            smmce_loss[j] *= (args.smmce_weight / np.sqrt(n))

            # Compute the regularization term over dataset j.
            cov_loss[j] = F.binary_cross_entropy_with_logits(
                logits[j], torch.ones_like(logits[j]), reduction='sum')
            cov_loss[j] *= (args.coverage_weight / n)

            # Optionally, compute the S-BCE AUC to use for computing the kappa-
            # worst datasets out of the batch to use for DRO-style optimization.
            if args.kappa > 0:
                with torch.no_grad():
                    sbce_aucs[j] = metrics.compute_metric_auc(
                        outputs=batch.confidences[j],
                        targets=batch.labels[j],
                        weights=weights[j],
                        metric_fn=metrics.compute_ece,
                        num_auc_values=10).auc

        # Optional: If we are using kappa-worst DRO, then we take the
        # kappa-worst batches. Otherwise, we take all batches.
        if args.kappa > 0:
            indices = torch.topk(sbce_aucs, args.kappa).indices
        else:
            indices = torch.arange(len(sbce_aucs), device=device)

        # Aggregate total loss.
        smmce_loss = torch.index_select(smmce_loss, 0, indices).mean()
        cov_loss = torch.index_select(cov_loss, 0, indices).mean()
        loss = smmce_loss + cov_loss

        # Update parameters.
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            selector.parameters(),
            args.clip_grad_norm,
            error_if_nonfinite=True)
        optimizer.step()

        # Update averages and log.
        loss_ema = loss_ema * 0.9 + float(loss) * 0.1
        smmce_ema = smmce_ema * 0.9 + float(smmce_loss) * 0.1
        cov_ema = cov_ema * 0.9 + float(cov_loss) * 0.1
        if i % args.print_freq == 0:
            logging.info(utils.format_train_metrics(
                iteration=i,
                total_iterations=len(data_loader),
                loss=loss_ema,
                smmce=smmce_ema,
                coverage=cov_ema))

    return selector


def evaluate_model(selector, data_loader):
    """Evaluate the selective calibration error AUC of the selector.

    Args:
        selector: SelectiveNet nn.Module implementing the selector g(X).
        data_loader: DataLoader for evaluation data, where each batch can be
            converted into a BatchedInputDataset.

    Returns:
        A tuple of (average-case, worst-case) selective calibration error AUC.
    """
    selector.eval()
    device = next(selector.parameters()).device
    all_outputs, all_targets, all_weights = [], [], []

    with torch.no_grad():
        for batch in data_loader:
            batch = BatchedInputDataset(*[ex.to(device) for ex in batch])

            # Forward pass.
            m, n = batch.labels.size()
            features_flat = batch.input_features.view(m * n, -1)
            logits = selector(features_flat.float()).view(m, n)
            weights = torch.sigmoid(logits)

            # Aggregate.
            all_outputs.append(batch.confidences.cpu())
            all_targets.append(batch.labels.cpu())
            all_weights.append(weights.cpu())

    # Compute metrics.
    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_weights = torch.cat(all_weights, dim=0)

    all_aucs = []
    for idx in range(len(all_outputs)):
        all_aucs.append(metrics.compute_metric_auc(
            outputs=all_outputs[idx],
            targets=all_targets[idx],
            weights=all_weights[idx],
            metric_fn=metrics.compute_ece).auc)

    logging.info(utils.format_eval_metrics(all_aucs))

    return 100 * np.mean(all_aucs), 100 * np.max(all_aucs)


# ------------------------------------------------------------------------------
#
# Main.
#
# ------------------------------------------------------------------------------


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.model_dir is None:
        args.model_dir = tempfile.mkdtemp()

    if utils.check_nonempty(args.model_dir):
        if not args.overwrite:
            raise RuntimeError(
                'model-dir already exists! Use --overwrite to continue.')
        else:
            shutil.rmtree(args.model_dir)
    os.makedirs(args.model_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(os.path.join(args.model_dir, 'log.txt')),
            logging.StreamHandler()])

    logging.info(f'Command: {" ".join(sys.argv[:])}')
    logging.info(f'Saving logs and models to {args.model_dir}')

    # Save args.
    torch.save(args, os.path.join(args.model_dir, 'args.pt'))

    # Set device.
    args.cuda = torch.cuda.is_available() and not args.use_cpu
    device = torch.device("cuda:0" if args.cuda else "cpu")

    # --------------------------------------------------------------------------
    #
    # Load data.
    #
    # --------------------------------------------------------------------------
    logging.info('Loading data...')
    cal_dataset = torch.utils.data.TensorDataset(*torch.load(args.cal_dataset))
    cal_loader = torch.utils.data.DataLoader(
        dataset=cal_dataset,
        batch_size=args.train_batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=args.cuda)

    val_dataset = torch.utils.data.TensorDataset(*torch.load(args.val_dataset))
    # data = torch.load(args.val_dataset)
    # data = [data[-1], data[2], data[0], data[1]]
    # val_dataset = torch.utils.data.TensorDataset(*data)
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=args.cuda)

    # --------------------------------------------------------------------------
    #
    # Initialize model and optimizer.
    #
    # --------------------------------------------------------------------------
    logging.info('Initializing model...')
    input_dim = cal_dataset.tensors[0].size(-1)
    logging.info(f'Input dimension is {input_dim}')

    selector = SelectiveNet(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout).to(device)

    optimizer = torch.optim.Adam(
        selector.parameters(),
        args.learning_rate,
        weight_decay=args.weight_decay)

    # --------------------------------------------------------------------------
    #
    # Train model.
    #
    # --------------------------------------------------------------------------
    # Save initial model.
    checkpoint_name = os.path.join(args.model_dir, 'model.pt')
    save_model(checkpoint_name, 0, selector, optimizer)
    logging.info("=" * 88)
    results = evaluate_model(selector, val_loader)
    best_metric = results[1] if args.kappa > 0 else results[0]
    logging.info(f'Current selective AUC: {best_metric:.3e}')

    # Train for n epochs, saving best model along the way.
    for epoch in range(1, args.epochs + 1):
        logging.info("=" * 88)
        logging.info(f'Starting epoch {epoch}/{args.epochs}...')
        selector = train_model(args, selector, cal_loader, optimizer)
        logging.info("=" * 88)
        results = evaluate_model(selector, val_loader)
        metric = results[1] if args.kappa > 0 else results[0]
        logging.info(f'Current selective AUC: {metric:.3e}')

        # Save if best.
        if metric < best_metric:
            logging.info('Best model so far '
                         f'({metric:.3e} vs. {best_metric:.3e}).')
            best_metric = metric
            save_model(checkpoint_name, epoch, selector, optimizer)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
