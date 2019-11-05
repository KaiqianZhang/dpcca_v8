"""=============================================================================
Train a deep variational CCA model.
============================================================================="""

import argparse
import time

import torch
import torch.utils.data
from   torch import optim
from   torch.nn import functional as F

import cuda
from   data import loader
from   models import DVCCA
import pprint

# ------------------------------------------------------------------------------

LOG_EVERY        = 1
SAVE_MODEL_EVERY = 100

mm     = torch.mm
device = cuda.device()

# ------------------------------------------------------------------------------

def main(args):
    """Main program: train -> test once per epoch while saving samples as
    needed.
    """
    start_time = time.time()

    pprint.set_logfiles(args.directory)
    pprint.log_section('Loading dataset and config.')

    pprint.log_section('Loading script arguments.')
    pprint.log_args(args)

    cfg = loader.get_config(args.dataset)

    train_loader, test_loader = loader.get_data_loaders(cfg,
                                                        args.batch_size,
                                                        args.n_workers,
                                                        args.pin_memory,
                                                        args.cv_pct)

    cfg.EMBEDDING_DIM = args.latent_dim
    model = DVCCA(cfg)

    model = model.to(device)

    pprint.log_section('Model specs.')
    pprint.log_model(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    pprint.log_section('Training model.')
    for epoch in range(1, args.n_epochs + 1):

        loss_tr = train(train_loader, model, optimizer)
        loss_te = test(cfg, args, epoch, test_loader, model)

        msg = '{}\t|\t{:6f}\t{:6f}'.format(epoch, loss_tr, loss_te)
        pprint.log(msg)

        if epoch % LOG_EVERY == 0:
            save_samples(args.directory, model, test_loader, cfg, epoch)

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

        x1.requires_grad_()
        x2.requires_grad_()

        x1 = x1.to(device)
        x2 = x2.to(device)
        x1r, x2r, mu, logvar = model.forward([x1, x2])
        loss = loss_fn(x1r, x2r, x1, x2, mu, logvar)

        loss.backward()
        optimizer.step()
        loss_sum += loss.item()

    loss_sum /= i
    return loss_sum

# ------------------------------------------------------------------------------

def test(cfg, args, epoch, test_loader, model):
    """Test model by computing the average loss on a held-out dataset. No
    parameter updates.
    """
    model.eval()
    loss_sum = 0

    for i, (x1, x2) in enumerate(test_loader):

        x1 = x1.to(device)
        x2 = x2.to(device)

        x1r, x2r, mu, logvar = model.forward([x1, x2])
        loss = loss_fn(x1r, x2r, x1, x2, mu, logvar)
        loss_sum += loss.item()

        if i == 0 and epoch % LOG_EVERY == 0:
            cfg.save_comparison(args.directory, x1, x1r, epoch, is_x1=True)
            cfg.save_comparison(args.directory, x2, x2r, epoch, is_x1=False)

    loss_sum /= i
    return loss_sum

# ------------------------------------------------------------------------------

def loss_fn(x1r, x2r, x1, x2, mu, logvar):
    mse1 = F.mse_loss(x1r, x1)
    mse2 = F.mse_loss(x2r, x2)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return mse1 + mse2 + kld

# ------------------------------------------------------------------------------

def save_samples(directory, model, test_loader, cfg, epoch):
    """Save samples from test set.
    """
    n  = len(test_loader.sampler.indices)
    x1_batch = torch.Tensor(n, cfg.N_CHANNELS, cfg.IMG_SIZE, cfg.IMG_SIZE)
    for i in range(n):
        j = test_loader.sampler.indices[i]
        x1, x2 = test_loader.dataset[j]
        x1_batch[i] = x1
    x1_batch = x1_batch.to(device)
    cfg.save_image_samples(directory, model, epoch, x1_batch)

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
    p.add_argument('--directory',   type=str,   default='experiments/test')
    p.add_argument('--wall_time',   type=int,   default=24)
    p.add_argument('--seed',        type=int,   default=0)

    p.add_argument('--dataset',     type=str,   default='gtex')
    p.add_argument('--batch_size',  type=int,   default=128)
    p.add_argument('--n_epochs',    type=int,   default=100)
    p.add_argument('--cv_pct',      type=float, default=0.1)
    p.add_argument('--lr',          type=float, default=1e-4)
    p.add_argument('--latent_dim',  type=int,   default=1000)

    args, _ = p.parse_known_args()

    args.n_workers  = 4
    args.pin_memory = torch.cuda.is_available()

    torch.manual_seed(args.seed)
    main(args)
