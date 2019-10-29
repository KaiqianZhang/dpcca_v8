"""=============================================================================
Train a variational autoencoder.
============================================================================="""

import argparse
import time

import torch
import torch.utils.data
from   torch import optim
from   torchvision.utils import save_image

import cuda
from   data import loader
import pprint
from   models import DCGANVAE128

# ------------------------------------------------------------------------------

def main(args):

    start_time = time.time()

    pprint.set_logfiles(args.directory)

    pprint.log_section('Loading config.')
    cfg = loader.get_config(args.dataset)
    pprint.log_config(cfg)

    pprint.log_section('Loading script arguments.')
    pprint.log_args(args)

    pprint.log_section('Loading dataset.')
    train_loader, test_loader = loader.get_data_loaders(cfg,
                                                        args.batch_size,
                                                        args.n_workers,
                                                        args.pin_memory,
                                                        cv_pct=0.1)

    if (args.batch_size / args.n_gpus) > len(train_loader.sampler.indices):
        raise ValueError('Train set must be greater than batch size.')
    if (args.batch_size / args.n_gpus) > len(test_loader.sampler.indices):
        raise ValueError('Test set must be greater than batch size.')

    model = DCGANVAE128(cfg)
    model = model.to(cuda.device())
    pprint.log_section('Model specs.')
    pprint.log_model(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    pprint.log_section('Training model.')
    for epoch in range(1, args.n_epochs + 1):
        train_msgs = train(args, train_loader, model, optimizer)
        test_msgs  = test(args, cfg, epoch, test_loader, model)
        pprint.log_line(epoch, train_msgs, test_msgs)

        if epoch % args.log_every == 0 and args.modality == 'images':
            save_sample(cfg, model, epoch)
        if epoch % args.log_every == 0:
            save_model(args, model)

    save_model(args, model)
    pprint.log_section('Model saved.')
    hours = round((time.time() - start_time) / 3600, 2)
    pprint.log_section('Job complete in %s hrs.' % hours)

# ------------------------------------------------------------------------------

def train(args, train_loader, model, optimizer):
    model.train()
    recon_loss = 0
    kld_loss   = 0

    for i, (x1, x2) in enumerate(train_loader):
        optimizer.zero_grad()

        x = x1 if args.modality == 'images' else x2
        x = cuda.ize(x)
        xr, mu, logvar = model.forward(x)
        recon, kld = model.loss(xr, x, mu, logvar)
        loss = recon + (args.beta * kld)
        loss.backward()
        optimizer.step()

        recon_loss += recon.item()
        kld_loss   += kld.item()

    recon_loss /= len(train_loader.sampler.indices)
    kld_loss   /= len(train_loader.sampler.indices)
    return [recon_loss, kld_loss]

# ------------------------------------------------------------------------------

def test(args, cfg, epoch, test_loader, model):
    model.eval()
    recon_loss = 0
    kld_loss   = 0

    with torch.no_grad():
        for i, (x1, x2) in enumerate(test_loader):

            x = x1 if args.modality == 'images' else x2
            x = cuda.ize(x)
            xr, mu, logvar = model.forward(x)
            recon, kld  = model.loss(xr, x, mu, logvar)
            recon_loss += recon.item()
            kld_loss   += kld.item()

            if i == 0 and epoch % args.log_every == 0 and args.modality == 'images':
                cfg.save_comparison(args.directory, x, xr, epoch, is_x1=True)

    recon_loss /= len(test_loader.sampler.indices)
    kld_loss   /= len(test_loader.sampler.indices)
    return [recon_loss, kld_loss]

# ------------------------------------------------------------------------------

def save_sample(cfg, model, epoch):
    N = 64
    with torch.no_grad():
        sample = torch.randn(N, cfg.IMG_EMBED_DIM, device=cuda.device())
        sample = model.decode(sample).cpu()[:N]
        image  = sample.view(N, cfg.N_CHANNELS, cfg.IMG_SIZE, cfg.IMG_SIZE)
        fname  = '%s/sample_%s.png' % (args.directory, str(epoch))
        save_image(image, fname)

# ------------------------------------------------------------------------------

def save_model(args, model):
    fpath = '%s/model.pt' % args.directory
    state = model.state_dict()
    torch.save(state, fpath)

# ------------------------------------------------------------------------------

if __name__ == '__main__':
    p = argparse.ArgumentParser()

    p.add_argument('--wall_time',  type=int,   default=24)
    p.add_argument('--n_gpus',     type=int,   default=1)
    p.add_argument('--modality',   type=str,   default='images')
    p.add_argument('--dataset',    type=str,   default='gtexv8')
    p.add_argument('--batch_size', type=int,   default=128)
    p.add_argument('--latent_dim', type=int,   default=2)
    p.add_argument('--lr',         type=float, default=1e-3)
    p.add_argument('--n_epochs',   type=int,   default=100)
    p.add_argument('--seed',       type=int,   default=0)
    p.add_argument('--cv_pct',     type=float, default=0.1)
    # See beta-VAE: https://openreview.net/pdf?id=Sy2fzU9gl.
    p.add_argument('--beta',       type=float, default=1.0)
    p.add_argument('--directory',  type=str,   default='experiments/test')

    args, _ = p.parse_known_args()

    args.n_workers  = 4 * args.n_gpus
    args.pin_memory = torch.cuda.is_available()
    args.log_every  = 1

    torch.manual_seed(args.seed)
    main(args)
