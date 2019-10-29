"""=============================================================================
Train cross-modality autoencoders.
============================================================================="""

import argparse
import matplotlib.pyplot as plt

import torch
from   torch import optim
from   torch.nn import functional as F
from   torchvision.utils import save_image

import cuda
from   data import loader
from   models import CrossModalityAE
import jobutils

# ------------------------------------------------------------------------------

def main(args):
    """Run full training and embedding program.
    """
    print('Using CUDA: %s.' % args.cuda)

    cfg   = loader.get_config(args.dataset)
    model = CrossModalityAE(cfg)
    model = cuda.ize(model)

    args.directory = make_subdir(args, model.__name__())

    train_loader, cv_loader = loader.get_data_loaders(cfg,
                                                      args.batch_size,
                                                      args.n_workers,
                                                      args.pin_memory,
                                                      cv_pct=0.1)
    print('Data loaded.')

    # Train model
    # --------------------------------------------------------------------------
    model = train(cfg, model, train_loader, args)
    print('Model trained.')

    # Save model
    # --------------------------------------------------------------------------
    fpath = '%s/model.pt' % args.directory
    state = model.state_dict()
    torch.save(state, fpath)
    print('Model saved.')

# ------------------------------------------------------------------------------

def train(cfg, model, train_loader, args):
    """Train model.
    """
    model.train()

    model     = cuda.ize(model)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for i in range(args.n_epochs):
        train_epoch(i, cfg, model, train_loader, optimizer)
    return model

# ------------------------------------------------------------------------------

def train_epoch(epoch, cfg, model, train_loader, optimizer):
    """Useful function to free up memory.
    """
    err_sum   = 0
    err11_sum = 0
    err12_sum = 0
    err21_sum = 0
    err22_sum = 0
    err3_sum  = 0

    for j, (x1, x2) in enumerate(train_loader):
        optimizer.zero_grad()

        x1 = x1.requires_grad_()
        x2 = x2.requires_grad_()
        x1 = cuda.ize(x1)
        x2 = cuda.ize(x2)

        z1, \
        z2, \
        x1_from_z1, \
        x1_from_z2, \
        x2_from_z2, \
        x2_from_z1 = model.forward([x1, x2])

        err11 = F.mse_loss(x1_from_z1, x1.detach())
        err12 = F.mse_loss(x1_from_z2, x1.detach())
        err21 = F.mse_loss(x2_from_z2, x2.detach())
        err22 = F.mse_loss(x2_from_z1, x2.detach())
        err3  = 4 * (1 - F.cosine_similarity(z1, z2)).sum()

        if j == 0:
            save_reconstructions(epoch, cfg, x2, x2_from_z2)
            save_reconstructions(epoch, cfg, x2, x2_from_z1, cross_mode=True)

        err = err11 + err12 + err21 + err22 + err3
        err.sum().backward()
        optimizer.step()

        err_sum   += err.item()
        err11_sum += err11.item()
        err12_sum += err12.item()
        err21_sum += err21.item()
        err22_sum += err22.item()
        err3_sum  += err3.item()

    msg = '{}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}'.format(
        epoch, err11_sum, err12_sum, err21_sum, err22_sum, err3_sum)

    print(msg)

# ------------------------------------------------------------------------------

def save_reconstructions(epoch, cfg, x, x_recon, cross_mode=False):
    fpath = '%s/%s_recon_%s.png' % (args.directory,
                                str(epoch),
                                'cm' if cross_mode else '')
    x = x.detach().numpy()
    x_recon = x_recon.detach().numpy()
    fig, ax = plt.subplots()
    ax.scatter(x[:, 0], x[:, 1], c='red', marker='.')
    ax.scatter(x_recon[:, 0], x_recon[:, 1], c='orange', marker='*')
    plt.savefig(fpath)
    plt.axes().set_aspect('equal', 'datalim')
    plt.close('all')
    plt.clf()
    # n = min(x.size(0), 8)
    # recon = x_recon.view(args.batch_size, cfg.N_CHANNELS, cfg.IMG_SIZE,
    #                      cfg.IMG_SIZE)
    # x = x.view(args.batch_size, cfg.N_CHANNELS, cfg.IMG_SIZE, cfg.IMG_SIZE)
    # comparison = torch.cat([x[:n], recon[:n]])
    # save_image(comparison.cpu(), fpath, nrow=n)

# ------------------------------------------------------------------------------

def make_subdir(args, model_name):
    """Ensure necessary directories exist before launching program.
    """
    directory = 'experiments/ae/%s' % model_name
    jobutils.mkdir(directory)
    jobutils.mkdir(directory)
    return directory

# ------------------------------------------------------------------------------

if __name__ == '__main__':
    p = argparse.ArgumentParser()

    # Experimental setup.
    p.add_argument('--dataset',      required=False, type=str,   default='gtex')
    p.add_argument('--n_epochs',     required=False, type=int,   default=1000)
    p.add_argument('--batch_size',   required=False, type=int,   default=128)
    p.add_argument('--lr',           required=False, type=float, default=0.001)
    p.add_argument('--n_z_per_samp', required=False, type=int,   default=100)

    # Job settings.
    p.add_argument('--use_cuda',     required=False, type=bool, default=True)
    p.add_argument('--n_workers',    required=False, type=int,  default=4)

    args = p.parse_args()

    # Put these values on `args` so their values are logged.
    args.cuda = args.use_cuda and torch.cuda.is_available()
    # Pinning memory only works when using GPUs.
    args.pin_memory = args.cuda

    main(args)
