"""=============================================================================
Train convolutional autoencoder to learn image embeddings.
============================================================================="""

import argparse

import torch
from   torch import optim
from   torch.nn import functional as F

import cuda
from   data import loader
from   models import MAE, LeNet5AE
import pprint

# ------------------------------------------------------------------------------

LOG_EVERY = 1

# ------------------------------------------------------------------------------

def main(args):
    """Run full training and embedding program.
    """
    pprint.set_logfiles(args.directory)

    pprint.log_section('Loading config.')
    cfg = loader.get_config(args.dataset)
    pprint.log_config(cfg)

    pprint.log_section('Loading script arguments.')
    pprint.log_args(args)

    pprint.log_section('Model specs.')
    model = MAE(cfg)

    pprint.log_section('Loading dataset.')
    train_loader, cv_loader = loader.get_data_loaders(cfg,
                                                      args.batch_size,
                                                      args.n_workers,
                                                      args.pin_memory,
                                                      cv_pct=0.1)

    # Train model
    # --------------------------------------------------------------------------
    pprint.log_section('Training.')
    model = train(cfg, model, train_loader, args)

    # Save model
    # --------------------------------------------------------------------------
    pprint.log_section('Saving model.')
    fpath = '%s/model.pt' % args.directory
    state = model.state_dict()
    torch.save(state, fpath)

# ------------------------------------------------------------------------------

def train(cfg, model, train_loader, args):
    """Train model.
    """
    model.train()

    model     = cuda.ize(model)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for i in range(args.n_epochs):
        train_epoch(args, i, cfg, model, train_loader, optimizer)
    return model

# ------------------------------------------------------------------------------

def train_epoch(args, epoch, cfg, model, train_loader, optimizer):
    """Useful function to free up memory.
    """
    err1_sum = 0
    err2_sum = 0

    for j, (x1, x2) in enumerate(train_loader):
        optimizer.zero_grad()

        x1 = x1.to(cuda.device())
        x2 = x2.to(cuda.device())

        x1r, x2r = model.forward([x1, x2])
        err1 = F.mse_loss(x1r, x1)
        err2 = F.mse_loss(x2r, x2)
        err  = err1 + err2

        err.sum().backward()
        optimizer.step()

        err1_sum += err1.item()
        err2_sum += err2.item()

        if j == 0 and epoch % LOG_EVERY == 0:
            cfg.save_comparison(args.directory, x1, x1r, epoch, is_x1=True)
            cfg.save_comparison(args.directory, x2, x2r, epoch, is_x1=False)

    err1_sum /= j
    err2_sum /= j

    msg = '%s\t%s\t%s' % (epoch, err1_sum, err2_sum)
    print(msg)

# ------------------------------------------------------------------------------

if __name__ == '__main__':
    p = argparse.ArgumentParser()

    # Experimental setup.
    p.add_argument('--dataset',
                   required=False,
                   type=str,
                   default='mnist')
    p.add_argument('--directory',
                   required=False,
                   type=str,
                   default='experiments/test')
    p.add_argument('--n_epochs',
                   required=False,
                   type=int,
                   default=500)
    p.add_argument('--batch_size',
                   required=False,
                   type=int,
                   default=128)
    p.add_argument('--lr',
                   required=False,
                   type=float,
                   default=0.001)
    p.add_argument('--modality',
                   required=False,
                   type=str,
                   default='images')

    args, _ = p.parse_known_args()

    args.cuda       = torch.cuda.is_available()
    args.n_workers  = 4 if args.cuda else 0
    args.pin_memory = args.cuda

    main(args)
