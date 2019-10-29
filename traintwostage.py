"""=============================================================================
Train deep probabilistic CCA in a two-stage approach.
============================================================================="""

import argparse
import time

import torch
import torch.utils.data
from   torch import optim
from   torch.nn import functional as F
from   torch.distributions.multivariate_normal import MultivariateNormal as MVN

import metrics
import cuda
from   data import loader
from   models import DPCCA, PCCAVec
import pprint
import jobutils

# ------------------------------------------------------------------------------

LOG_EVERY        = 10
SAVE_MODEL_EVERY = 100

diag   = torch.diag
mm     = torch.mm
device = cuda.device()

# ------------------------------------------------------------------------------

def main(args):
    """Main program: train -> test once per epoch while saving samples as
    needed.
    """
    start_time = time.time()

    pprint.set_logfile('%s/out.txt' % args.directory)
    pprint.log_section('Loading dataset and config.')

    pprint.log_section('Loading script arguments.')
    pprint.log_args(args)

    cfg = loader.get_config(args.dataset)

    train_loader, test_loader = loader.get_data_loaders(cfg,
                                                        args.batch_size,
                                                        args.n_workers,
                                                        args.pin_memory,
                                                        args.cv_pct)

    pprint.log_section('Pretrain image autoencoder.')
    image_net = pretrain_image_net(args, cfg, train_loader)

    pprint.log_section('Pretrain genes autoencoder.')
    genes_net = pretrain_genes_net(args, cfg, train_loader)

    pprint.log_section('Fitting PCCA.')
    pcca = fit_pcca(args, cfg, train_loader, image_net, genes_net)

    # This is useful because we can treat this two-model like any other.
    model = DPCCA(cfg, args.latent_dim, private_z=True)
    model.genes_net = genes_net
    model.image_net = image_net
    model.pcca      = pcca

    model.eval()

    pprint.log_section('Sample from PCCA.')
    sample(cfg, args, train_loader, image_net, genes_net, pcca)

    pprint.log_section('Analyzing model.')
    dataset = train_loader.dataset
    metrics.log_metrics(model, dataset)

    hours = round((time.time() - start_time) / 3600, 1)
    pprint.log_section('Job complete in %s hrs.' % hours)

    jobutils.save_model(args.directory, model)
    pprint.log_section('Model saved.')

# ------------------------------------------------------------------------------

def fit_pcca(args, cfg, train_loader, image_net, genes_net):
    """
    """
    pcca = PCCAVec(latent_dim=args.latent_dim,
                   dims=[cfg.IMG_EMBED_DIM, cfg.GENE_EMBED_DIM],
                   n_iters=1)

    x1 = train_loader.dataset.images.to(device)
    x2 = train_loader.dataset.genes.to(device)

    y1 = image_net.encode(x1)
    y2 = genes_net.encode(x2)

    # PCCA assumes our data is mean-centered.
    y1 = y1 - y1.mean(dim=0)
    y2 = y2 - y2.mean(dim=0)

    y = torch.cat([y1, y2], dim=1)

    # PCCA expects (p-dims, n-samps)-dimensional data.
    y = y.t()

    pcca.forward(y)
    return pcca

# ------------------------------------------------------------------------------

def pretrain_genes_net(args, cfg, train_loader):
    """
    """
    model = cfg.get_genes_net(**{'input_dim': cfg.N_GENES})
    model.train()
    model     = cuda.ize(model)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.n_epochs):
        _pretrain(epoch, model, train_loader, optimizer, False)
    return model

# ------------------------------------------------------------------------------

def pretrain_image_net(args, cfg, train_loader):
    """
    """
    model = cfg.get_image_net(**{'input_dim': cfg.N_PIXELS})
    model.train()
    model     = cuda.ize(model)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.n_epochs):
        _pretrain(epoch, model, train_loader, optimizer, True)
    return model

# ------------------------------------------------------------------------------

def _pretrain(epoch, model, train_loader, optimizer, is_x1):
    """Train gene-net autoencoder.
    """
    err_sum = 0
    for j, (x1, x2) in enumerate(train_loader):

        x = x1 if is_x1 else x2

        optimizer.zero_grad()

        x.requires_grad_()
        x = x.to(device)

        xr = model.forward(x)
        err = F.mse_loss(xr, x.detach())

        err.sum().backward()
        err_sum += err.item()
        optimizer.step()
        #
        # if j == 0 and epoch % LOG_EVERY == 0:
        #     cfg.save_comparison(args.directory, x, xr, epoch, is_x1)

    msg = '%s\t%s' % (epoch, err_sum)
    print(msg)

# ------------------------------------------------------------------------------

def sample(cfg, args, train_loader, image_net, genes_net, pcca, n_samples=200):
    """
    """
    x1 = train_loader.dataset.images.to(device)
    x2 = train_loader.dataset.genes.to(device)

    pprint.log('Encoding data.')
    y1 = image_net.encode(x1)
    y2 = genes_net.encode(x2)

    y1 = y1 - y1.mean(dim=0)
    y2 = y2 - y2.mean(dim=0)
    y = torch.cat([y1, y2], dim=1)
    y = y.t()

    pprint.log('Estimating Z.')
    z_est = pcca.estimate_z_given_y(y)

    k = pcca.latent_dim
    zs_est = z_est[:k, :]
    z1_est = z_est[k:2 * k, :]
    z2_est = z_est[2 * k:, :]
    m1 = mm(pcca.Lambda1, zs_est) + mm(pcca.B1, z1_est)
    m2 = mm(pcca.Lambda2, zs_est) + mm(pcca.B2, z2_est)

    pprint.log('Sampling %s data points' % n_samples)
    y1_emb_est = torch.zeros((n_samples, pcca.p1))
    y2_emb_est = torch.zeros((n_samples, pcca.p2))

    for i in range(n_samples):
        y1_emb_est[i, :] = MVN(m1[:, i], diag(pcca.Psi1)).sample()
        y2_emb_est[i, :] = MVN(m2[:, i], diag(pcca.Psi2)).sample()

    y1_emb_est = y1_emb_est.to(cuda.device())
    y2_emb_est = y2_emb_est.to(cuda.device())

    pprint.log('Decoding data.')
    x1r = image_net.decode(y1_emb_est)
    x2r = genes_net.decode(y2_emb_est)

    pprint.log(x2.shape)
    pprint.log(x2r.shape)

    x1  = x1[:n_samples]
    x1r = x1r[:n_samples]
    x2  = x2[:n_samples]
    x2r = x2r[:n_samples]

    cfg.save_comparison(args.directory, x1, x1r, 'final', is_x1=True)
    cfg.save_comparison(args.directory, x2, x2r, 'final', is_x1=False)

# ------------------------------------------------------------------------------

if __name__ == '__main__':

    p = argparse.ArgumentParser()
    p.add_argument('--directory',   type=str,   default='experiments/test')
    p.add_argument('--n_gpus',      type=int,   default=1)
    p.add_argument('--wall_time',   type=int,   default=24)
    p.add_argument('--seed',        type=int,   default=0)

    p.add_argument('--dataset',     type=str,   default='gtexv8')
    p.add_argument('--batch_size',  type=int,   default=128)
    p.add_argument('--n_epochs',    type=int,   default=100)
    p.add_argument('--cv_pct',      type=float, default=0.1)
    p.add_argument('--lr',          type=float, default=1e-4)
    p.add_argument('--latent_dim',  type=int,   default=5)
    p.add_argument('--l1_coef',     type=float, default=0.0)
    p.add_argument('--private_z',   type=int,   default=0)

    args, _ = p.parse_known_args()

    args.n_workers  = 4 * args.n_gpus
    args.pin_memory = torch.cuda.is_available()
    args.private_z  = bool(args.private_z)

    torch.manual_seed(args.seed)
    main(args)
