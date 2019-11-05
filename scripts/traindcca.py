"""=============================================================================
Train a deep CCA model.
============================================================================="""

import argparse
import time

import torch
import torch.utils.data
from   torch import optim

import cuda
from   data import loader
from   models import DCCA
import pprint

# ------------------------------------------------------------------------------

LOG_EVERY = 1
SAVE_MODEL_EVERY = 100

mm = torch.mm
device = cuda.device()

# ------------------------------------------------------------------------------

def main(args):
    """Main program: train -> test once per epoch while saving samples as
    needed.
    """
    start_time = time.time()

    pprint.set_logfiles(args.directory)
    pprint.log_section('Loading dataset and config.')

    if args.full_batch != 0:
        args.batch_size = -1

    pprint.log_section('Loading script arguments.')
    pprint.log_args(args)

    cfg = loader.get_config(args.dataset)

    train_loader, test_loader = loader.get_data_loaders(cfg,
                                                        args.batch_size,
                                                        args.n_workers,
                                                        args.pin_memory,
                                                        args.cv_pct)

    cfg.IMG_EMBED_DIM  = args.latent_dim
    cfg.GENE_EMBED_DIM = args.latent_dim
    model = DCCA(cfg)
    model = model.to(device)

    pprint.log_section('Model specs.')
    pprint.log_model(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    pprint.log_section('Training model.')
    for epoch in range(1, args.n_epochs + 1):

        loss, grad = train(train_loader, model, optimizer)
        msg = '{}\t|\t{:6f}\t{:6f}'.format(epoch, loss, grad)
        pprint.log(msg)

        if epoch % SAVE_MODEL_EVERY == 0:
            save_model(args.directory, model)

    hours = round((time.time() - start_time) / 3600, 1)
    pprint.log_section('Job complete in %s hrs.' % hours)

    save_model(args.directory, model)
    pprint.log_section('Model saved.')

# ------------------------------------------------------------------------------

def train(train_loader, model, optimizer):
    """Train PCCA model and update parameters in batches of the whole train set.
    """
    model.train()
    loss_sum = 0

    for i, (x1, x2) in enumerate(train_loader):

        optimizer.zero_grad()
        x1 = x1.requires_grad_().to(device)
        x2 = x2.requires_grad_().to(device)

        h1, h2   = model.forward([x1, x2])
        neg_corr = model.cca_loss(h1, h2)

        loss = neg_corr.sum()
        loss.backward()
        loss_sum += loss.item()

        optimizer.step()

    return loss_sum

# ------------------------------------------------------------------------------

def save_model(directory, model):
    """Save PyTorch model's state dictionary for provenance.
    """
    fpath = '%s/model.pt' % directory
    state = model.state_dict()
    torch.save(state, fpath)

# ------------------------------------------------------------------------------

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--directory', type=str, default='experiments/test')
    p.add_argument('--wall_time', type=int, default=24)
    p.add_argument('--seed', type=int, default=0)

    p.add_argument('--dataset', type=str, default='gtex')
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--n_epochs', type=int, default=100)
    p.add_argument('--cv_pct', type=float, default=0.1)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--latent_dim', type=int, default=1000)

    # If 1, train using entire dataset in one go.
    p.add_argument('--full_batch',  type=int,   default=0)

    args, _ = p.parse_known_args()

    args.n_workers = 4
    args.pin_memory = torch.cuda.is_available()

    # For easy debugging locally.
    if args.directory == 'experiments/test':
        LOG_EVERY = 1
        SAVE_MODEL_EVERY = 5

    torch.manual_seed(args.seed)
    main(args)
