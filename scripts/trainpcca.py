"""=============================================================================
Train pCCA on a multimodal dataset.
============================================================================="""

import argparse
import time

import torch
from   torch import optim
from   torch.nn import functional as F
import torch.utils.data
from   torchvision.utils import save_image

import cuda
from   data import loader
from   models import DPCCA
import pprint

# ------------------------------------------------------------------------------

LOG_EVERY  = 1

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
    cfg.PCCA_Z_DIM    = args.latent_dim
    cfg.PCCA_ALPHA    = args.pcca_alpha
    cfg.PCCA_EM_ITERS = 100

    train_loader, test_loader = loader.get_data_loaders(cfg,
                                                        128,
                                                        1,
                                                        False,
                                                        args.cv_pct)

    cfg.visualize_dataset(args.directory)

    pprint.log_section('Data loaded.')
    model = cuda.ize(DPCCA(cfg))

    if args.pretrain_aes:
        optimizer = optim.Adam(model.parameters(), lr=1e-2)
        model = pretrain(train_loader, model, optimizer,
                         n_pretraining_epochs=1000)
        pprint.log_section('Autoencoders pretrained.')
        x1e, x2e = embed_data(cfg, model, train_loader)
    else:
        x1e = cuda.ize(train_loader.dataset.images)
        x2e = cuda.ize(train_loader.dataset.genes)

    pprint.log_section('Data for PCCA has shapes:')
    print(x1e.shape)
    print(x2e.shape)

    pprint.log_section('Fitting PCCA model.')

    model = fit_pcca(x1e, x2e, model)
    pprint.log_section('PCCA model fit.')

    save_model(args, model)
    pprint.log_section('Model saved.')

    z_est = estimate_z(args.pretrain_aes, model, train_loader.dataset)
    z     = train_loader.dataset.z
    dist  = torch.norm(z - z_est).item()
    pprint.log_section('Latent variable estimated. Distance is: %s' % dist)

    if args.pretrain_aes:
        save_ae_samples(args, cfg, model, train_loader)
        cfg.save_samples(args.directory, model, 'full')
    else:
        cfg.save_samples(args.directory, model.pcca, 'full')

    hours = round((time.time() - start_time) / 3600, 1)
    pprint.log_section('Job complete in %s hrs.' % hours)

# ------------------------------------------------------------------------------

def pretrain(train_loader, model, optimizer, n_pretraining_epochs):
    """Pretrain autoencoders.
    """
    model.train()
    err1 = 0
    err2 = 0

    for epoch in range(n_pretraining_epochs):
        for i, (x1, x2) in enumerate(train_loader):

            optimizer.zero_grad()

            x1 = cuda.ize(x1)
            x2 = cuda.ize(x2)

            x1r = model.image_net.forward(x1)
            x2r = model.genes_net.forward(x2)

            recon1 = F.mse_loss(x1r, x1.detach())
            recon2 = F.mse_loss(x2r, x2)
            loss   = recon1 + recon2

            loss.backward()

            err1 += recon1.item()
            err2 += recon2.item()

            optimizer.step()

        err1 /= i
        err2 /= i

        msg = '{}\t{:.8f}\t{:.8f}'.format(epoch, err1, err2)
        pprint.log(msg)

    return model

# ------------------------------------------------------------------------------

def embed_data(cfg, model, train_loader):
    """Embed both modalities.
    """
    n_pixels = cfg.N_PIXELS

    x1 = train_loader.dataset.images.view(-1, n_pixels)
    x2 = train_loader.dataset.genes

    x1 = cuda.ize(x1)
    x2 = cuda.ize(x2)

    x1e = model.image_net.encode(x1)
    x2e = model.genes_net.encode(x2)
    return x1e, x2e

# ------------------------------------------------------------------------------

def fit_pcca(x1e, x2e, model):
    """Train PCCA model and update parameters in batches of the whole train set.
    """
    _, p1 = x1e.shape
    _, p2 = x2e.shape

    xe = torch.cat([x1e, x2e], dim=1)
    # We expect the data to be (p, n) and mean-centered.
    xe = xe - xe.mean(dim=0)
    xe = xe.t()

    _  = model.pcca.forward(xe, [p1, p2])

    return model

# ------------------------------------------------------------------------------

def save_ae_samples(args, cfg, model, train_loader):
    """Save image reconstruction samples.
    """
    x1  = cuda.ize(train_loader.dataset.images)
    x2  = cuda.ize(train_loader.dataset.genes)

    x1r = model.image_net.decode(model.image_net.encode(x1))
    x2r = model.image_net.decode(model.genes_net.encode(x2))

    cfg.save_comparison(args, x1, x1r, 'ae', True)
    cfg.save_comparison(args, x2, x2r, 'ae', False)

# ------------------------------------------------------------------------------

def save_model(args, model):
    """Save PyTorch model's state dictionary for provenance.
    """
    fpath = '%s/model.pt' % args.directory
    state = model.state_dict()
    torch.save(state, fpath)

# ------------------------------------------------------------------------------

def estimate_z(pretrain_aes, model, dataset):
    x1 = dataset.images
    x2 = dataset.genes
    if pretrain_aes:
        x1_emb = model.image_net.encode(x1)
        x2_emb = model.genes_net.encode(x2)
    else:
        x1_emb = x1
        x2_emb = x2
    x_emb  = torch.cat([x1_emb, x2_emb], dim=1).t()

    L = model.pcca.Lambda
    P = model.pcca.Psi
    z = model.pcca.E_z_given_x(L, P, x_emb)
    return z

# ------------------------------------------------------------------------------

if __name__ == '__main__':

    p = argparse.ArgumentParser()
    p.add_argument('--dataset',      type=str,   default='gtex')
    p.add_argument('--directory',    type=str,   default='test')
    p.add_argument('--wall_time',    type=int,   default=24)
    p.add_argument('--cv_pct',       type=float, default=0.1)

    p.add_argument('--pretrain_aes', type=int,   default=1)
    p.add_argument('--latent_dim',   type=int,   default=2)
    p.add_argument('--pcca_every',   type=int,   default=2)
    p.add_argument('--pcca_alpha',   type=float, default=1e-1)

    args, _ = p.parse_known_args()

    main(args)
