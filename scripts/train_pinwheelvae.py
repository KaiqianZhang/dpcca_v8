"""=============================================================================
Train VAE on pinwheel dataset.
============================================================================="""

import argparse
import numpy as np
import os

import torch
from   torch.autograd import Variable
from   torch import nn, optim
from   torch.nn import functional as F
from   torch.utils.data.sampler import SubsetRandomSampler
from   torch.utils.data import DataLoader
from   torch.utils.data import Dataset

import cuda
from   models import PinwheelAE, PinwheelVAE
from   data import pinwheel

# ------------------------------------------------------------------------------

DIRECTORY = 'data/pinwheel'

# ------------------------------------------------------------------------------

class PinwheelDataset(Dataset):
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i]

# ------------------------------------------------------------------------------

def train(data_loader, model, optimizer):
    model.train()

    err_sum = 0
    recon_sum = 0
    kld_sum = 0

    for x in data_loader:
        optimizer.zero_grad()

        x = cuda.ize(Variable(x))
        x_recon, mu, logvar = model.forward(x)
        # x_recon = model.forward(x)

        recon_loss = F.mse_loss(x_recon, x)
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        err_comb = recon_loss + kld_loss
        # err_comb = recon_loss

        err_sum += err_comb.data.mean()
        recon_sum += recon_loss.data.mean()
        kld_sum += kld_loss.data.mean()

        err_comb.sum().backward()
        optimizer.step()

    err_sum /= len(data_loader.dataset)
    recon_sum /= len(data_loader.dataset)
    kld_sum /= len(data_loader.dataset)
    return err_sum, recon_sum, kld_sum

# ------------------------------------------------------------------------------

def main(args):
    samples_per_cluster = int(args.n_samples / args.k_clusters)
    X = pinwheel.generate(args.k_clusters, samples_per_cluster)
    X = torch.Tensor(X)

    torch.save(X, '%s/train.pth' % DIRECTORY)

    dataset = PinwheelDataset(X)
    model   = PinwheelVAE()

    optimizer = optim.SGD(model.parameters(), lr=1e-3)

    train_indices = list(range(len(dataset)))
    train_loader = DataLoader(
        dataset,
        sampler=SubsetRandomSampler(train_indices),
        batch_size=50,
        num_workers=4,
        drop_last=True,
        pin_memory=args.use_cuda
    )

    out = open('%s/out.txt' % DIRECTORY, 'w+')
    for epoch in range(args.n_iters):
        # err, recon = train(train_loader, model, optimizer)
        # msg = '%s\t%s\t%s' % (epoch, err, recon)
        err, recon, kld = train(train_loader, model, optimizer)
        msg = '%s\t%s\t%s\t%s' % (epoch, err, recon, kld)
        if epoch % 100 == 0:
            out.write(msg + '\n')
            print(msg)
    out.close()

    fpath = '%s/model.pt' % DIRECTORY
    state = model.state_dict()
    torch.save(state, fpath)

# ------------------------------------------------------------------------------

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--lr', required=False, type=float, default=1e-3)
    p.add_argument('--out', required=False, type=str, default='')
    p.add_argument('--n_iters', required=False, type=int, default=1000)
    p.add_argument('--k_clusters', required=False, type=int, default=5)
    p.add_argument('--n_samples', required=False, type=int, default=5000)
    p.add_argument('--use_cuda', required=False, type=bool, default=False)
    args = p.parse_args()

    args.pin_memory = args.use_cuda and torch.cuda.is_available()
    if args.out:
        DIRECTORY = DIRECTORY + '/' + args.out
        if not os.path.isdir(DIRECTORY):
            os.system('mkdir %s' % DIRECTORY)

    main(args)
