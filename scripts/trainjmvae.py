"""=============================================================================
Train a joint multimodal variational autoencoder.
============================================================================="""

import argparse
import time

import torch
import torch.utils.data
from   torch import optim
from   torchvision.utils import save_image

import cuda
from   data import loader
from   models import JMVAE
import pprint
import plotutils

# ------------------------------------------------------------------------------

LOG_EVERY = 50

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
    cfg.LATENT_DIM = args.latent_dim

    train_loader, test_loader = loader.get_data_loaders(cfg,
                                                        args.batch_size,
                                                        args.n_workers,
                                                        args.pin_memory,
                                                        args.cv_pct)
    model = JMVAE(cfg)

    pprint.log_section('Model specs.')
    pprint.log_model(model)

    model = cuda.ize(model)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    pprint.log_section('Training model.')
    for epoch in range(1, args.n_epochs + 1):

        tr_recon1, tr_recon2, tr_kld = train(cfg, train_loader, model, optimizer)
        te_recon1, te_recon2, te_kld = test(cfg, epoch, test_loader, model)

        msg = '{}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}'.format(
            epoch, tr_recon1, tr_recon2, tr_kld, te_recon1, te_recon2, te_kld)
        pprint.log(msg)

        if epoch % LOG_EVERY == 0:
            save_image_sample(cfg, model, epoch)

        if epoch % LOG_EVERY == 0:
            save_model(args, model)

    save_model(args, model)
    pprint.log_section('Model saved.')

    hours = round((time.time() - start_time) / 3600, 1)
    pprint.log_section('Job complete in %s hrs.' % hours)

# ------------------------------------------------------------------------------

def train(cfg, train_loader, model, optimizer):
    model.train()
    recon1_loss = 0
    recon2_loss = 0
    kld_loss    = 0

    for i, (x1, x2) in enumerate(train_loader):
        optimizer.zero_grad()

        x1 = cuda.ize(x1)
        x2 = cuda.ize(x2)

        x1_recon, x2_recon, mu, logvar = model.forward([x1, x2])
        recon1, recon2, kld = model.loss(x1, x1_recon, x2, x2_recon, mu, logvar)

        # Normalize the losses by number of features. Otherwise, the loss for
        # images might be way higher simply because of the number of components.
        recon1 /= cfg.N_PIXELS
        recon2 /= cfg.N_GENES

        loss = recon1 + recon2 + kld

        recon1_loss += recon1.detach().item()
        recon2_loss += recon2.detach().item()
        kld_loss    += kld.detach().item()

        loss.backward()
        optimizer.step()

    recon1_loss /= len(train_loader.sampler.indices)
    recon2_loss /= len(train_loader.sampler.indices)
    kld_loss    /= len(train_loader.sampler.indices)
    return recon1_loss, recon2_loss, kld_loss

# ------------------------------------------------------------------------------

def test(cfg, epoch, test_loader, model):
    model.eval()
    recon1_loss = 0
    recon2_loss = 0
    kld_loss    = 0

    with torch.no_grad():
        for i, (x1, x2) in enumerate(test_loader):

            x1 = cuda.ize(x1)
            x2 = cuda.ize(x2)
            x1_recon, x2_recon, mu, logvar = model.forward([x1, x2])
            recon1, recon2, kld = model.loss(x1, x1_recon, x2, x2_recon, mu,
                                             logvar)

            recon1 /= cfg.N_PIXELS
            recon2 /= cfg.N_GENES

            recon1_loss += recon1.item()
            recon2_loss += recon2.item()
            kld_loss    += kld.item()

            if i == 0 and epoch % LOG_EVERY == 0:
                save_recon_comparison(cfg, epoch, x1, x1_recon, x2, x2_recon)

    recon1_loss /= len(test_loader.sampler.indices)
    recon2_loss /= len(test_loader.sampler.indices)
    kld_loss    /= len(test_loader.sampler.indices)
    return recon1_loss, recon2_loss, kld_loss

# ------------------------------------------------------------------------------

def save_recon_comparison(cfg, epoch, x1, x1_recon, x2, x2_recon):
    x1_fpath = '%s/%s_images_recon.png' % (args.directory, str(epoch))
    N = min(x1.size(0), 8)
    recon = x1_recon.view(-1, cfg.N_CHANNELS, cfg.IMG_SIZE, cfg.IMG_SIZE)[:N]
    comparison = torch.cat([x1[:N], recon])
    save_image(comparison.cpu(), x1_fpath, nrow=N)
    # x2_fpath = '%s/%s_genes_org.png' % (args.directory, str(epoch))
    # plotutils.corrmat(x2.cpu().numpy(), x2_fpath)
    # x2_fpath = '%s/%s_genes_recon.png' % (args.directory, str(epoch))
    # plotutils.corrmat(x2_recon.cpu().numpy(), x2_fpath)

# ------------------------------------------------------------------------------

def save_image_sample(cfg, model, epoch):
    """Save image samples from learned image likelihood.
    """
    with torch.no_grad():
        z = torch.randn(args.batch_size, cfg.LATENT_DIM)
        z = cuda.ize(z)
        x1_recon = model.image_net.decode(z)
        N = 64
        x1_recon = x1_recon[:N]
        x1_recon = x1_recon.view(N, cfg.N_CHANNELS, cfg.IMG_SIZE, cfg.IMG_SIZE)
        fname    = '%s/sample_%s.png' % (args.directory, str(epoch))
        save_image(x1_recon.cpu(), fname)

# ------------------------------------------------------------------------------

def save_model(args, model):
    fpath = '%s/model.pt' % args.directory
    state = model.state_dict()
    torch.save(state, fpath)

# ------------------------------------------------------------------------------

if __name__ == '__main__':

    p = argparse.ArgumentParser()
    p.add_argument('--wall_time',  type=int,   default=24)
    p.add_argument('--dataset',    type=str,   default='gtex')
    p.add_argument('--batch_size', type=int,   default=128)
    p.add_argument('--latent_dim', type=int,   default=2)
    p.add_argument('--lr',         type=float, default=1e-3)
    p.add_argument('--n_epochs',   type=int,   default=100)
    p.add_argument('--seed',       type=int,   default=0)
    p.add_argument('--cv_pct',     type=float, default=0.1)
    p.add_argument('--directory',  type=str,   default='experiments')
    args = p.parse_args()

    args.cuda       = torch.cuda.is_available()
    args.n_workers  = 4 if args.cuda else 1
    args.pin_memory = args.cuda

    torch.manual_seed(args.seed)
    main(args)
