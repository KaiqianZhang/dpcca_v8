"""=============================================================================
Train DMCM model.
============================================================================="""

import argparse
import gc
import time

import torch
from   torch import nn, optim
from   torch.autograd import Variable
from   torch.nn import functional as F

import cuda
import pprint
from   data import loader
from   models import DMCM
from scripts import embedder


# ------------------------------------------------------------------------------

def main(args):
    """Run full training program.
    """
    start_time = time.time()

    # Initialize logging.
    pprint.set_logfile('%s/out.txt' % args.directory)

    pprint.log_section('Loading dataset and config.')

    cfg = loader.get_config(args.dataset)
    train_loader, cv_loader = loader.get_data_loaders(cfg,
                                                      args.batch_size,
                                                      args.n_workers,
                                                      args.pin_memory,
                                                      args.cv_pct)
    pprint.log_config(cfg, args.mode)

    pprint.log_section('Loading script arguments.')
    pprint.log_args(args)

    avail = cuda.set_preference(args.use_cuda)
    pprint.log_section('Setting CUDA availability to: %s' % avail)

    pprint.log_section('Initializing model.')
    model = DMCM(args.mode, cfg)

    pprint.log_section('Training model.')
    train(model, train_loader, args)

    pprint.log_section('Saving model.')
    save_model(args, model)

    # This is the critical duration we need. Embedding can be done post-job.
    hours = round((time.time() - start_time) / 3600, 1)
    pprint.log_section('Job complete in %s hrs.' % hours)

    # Free anything we can before embedding.
    gc.collect()

    pprint.log_section('Embedding data.')
    embedder.preembed_images(args.directory, cfg, model, args.mode)


# ------------------------------------------------------------------------------

def train(model, train_loader, args):
    """Train model on given dataset for specificed number of epochs.
    """
    optimizer  = optim.Adam(model.parameters(), lr=args.lr)
    model      = cuda.ize(model)
    cos_sim    = cuda.ize(nn.CosineSimilarity())
    # Use MSE loss for autoencoder.
    recon_loss = cuda.ize(nn.MSELoss())

    for i in range(args.n_epochs):
        err_same, err_diff = train_one_epoch(args, model, train_loader,
                                             optimizer, cos_sim, recon_loss)
        msg = '%s\t%s\t%s' % (i, err_same, err_diff)
        pprint.log(msg)


# ------------------------------------------------------------------------------

def train_one_epoch(args, model, train_loader, optimizer, cos_sim, recon_loss):
    """Train model for one pass over the dataset.
    """
    model.train()

    err_same_sum = 0
    err_diff_sum = 0

    for (x1, x2), (x1d, x2d) in train_loader:

        # Gradients are accumulated in-place. We must zero them out each time.
        optimizer.zero_grad()

        # Mean center data for each feature. We can do this before turning the
        # data into autograd Variables because we don't care about computing the
        # gradient of this computation.
        x1  = cuda.ize(Variable(x1 - x1.mean(dim=0)))
        x2  = cuda.ize(Variable(x2 - x2.mean(dim=0)))
        x1d = cuda.ize(Variable(x1d - x1d.mean(dim=0)))
        x2d = cuda.ize(Variable(x2d - x2d.mean(dim=0)))

        # ======================================================================
        # Optimized CCA: paired samples should be similar.
        # ----------------------------------------------------------------------
        if args.mode == 'cca':
            z1, z2   = model.forward([x1, x2])
            err_same = 1 - cos_sim(z1, z2)
        elif args.mode == 'cca_ae':
            (z1, x1_recon), z2 = model.forward([x1, x2])
            err_same  = 1 - cos_sim(z1, z2)
            err_same += recon_loss(x1_recon, x1)
        elif args.mode == 'cca_vae':
            (z1, x1_recon, x1_mu, x1_logvar), z2 = model.forward([x1, x2])
            err_same  = 1 - cos_sim(z1, z2)
            err_same += vae_loss(x1_recon, x1, x1_mu, x1_logvar,
                                 model.conv_net.n_pixels,
                                 F.binary_cross_entropy)

        # err_same += l1_regularize(model.sparse_net, args.l1_coef)
        err_same.sum().backward()
        err_same_sum += err_same.data.mean()

        # ======================================================================
        # Handle dissimilarity if desired.
        # ----------------------------------------------------------------------
        if args.mode == 'cca':
            z1d, z2d = model.forward([x1d, x2d])
        elif args.mode == 'cca_ae':
            (z1d, _), z2d = model.forward([x1d, x2d])
        elif args.mode == 'cca_vae':
            (z1d, _, _, _), z2d = model.forward([x1d, x2d])
        err_diff = cos_sim(z1d, z2d)
        err_diff.sum().backward()
        err_diff_sum = err_diff.data.mean()

        # ======================================================================

        # Update model parameters.
        optimizer.step()

    err_same_sum /= len(train_loader.sampler.indices)
    err_diff_sum /= len(train_loader.sampler.indices)
    return err_same_sum, err_diff_sum


# ------------------------------------------------------------------------------

def l1_regularize(net, l1_coef):
    """Add L1 regularization.
    """
    # This code is nearly a copy of Soumith's:
    #
    #     https://discuss.pytorch.org/t/7951/2
    #
    l1_reg = None
    for param in net.parameters():
        if l1_reg is None:
            l1_reg = param.norm(1)
        else:
            l1_reg = l1_reg + param.norm(1)
    return l1_coef * l1_reg


# ------------------------------------------------------------------------------

def vae_loss(x_recon, x, mu, logvar, dim, recon_fn):
    """Compute VAE loss, which is a combination of reconstruction error and
    KL-divergence.
    """
    recon_loss = recon_fn(x_recon, x.view(-1, dim), size_average=False)

    # See Appendix B:
    #
    #     https://arxiv.org/abs/1312.6114
    #
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld_loss


# ------------------------------------------------------------------------------

def save_model(args, model):
    """Save model's state dict.
    """
    # The PyTorch documentation recommends saving the model's state dictionary
    # rather than saving the whole model:
    #
    #     https://github.com/pytorch/
    #         pytorch/blob/master/docs/source/notes/serialization.rst
    #
    fpath = '%s/model.pt' % args.directory
    state = model.state_dict()
    torch.save(state, fpath)


# ------------------------------------------------------------------------------

if __name__ == '__main__':
    p = argparse.ArgumentParser()

    # Experimental setup.
    p.add_argument('--directory',  required=True,  type=str)
    p.add_argument('--dataset',    required=True,  type=str)
    p.add_argument('--mode',       required=True,  type=str,   default='cca')
    p.add_argument('--n_epochs',   required=True,  type=int)
    p.add_argument('--batch_size', required=False, type=int,   default=128)
    p.add_argument('--cv_pct',     required=False, type=float, default=0.0)

    # This argument defines how unpaired samples are pushed away from each
    # other:
    #
    # - ''   --> No dissimilarity constraint
    # - 'i'  --> Image dissimilarity constraint
    # - 'g'  --> Gene dissimilarity constraint
    # - 'ig' --> Both image and gene dissimilarity constraint
    #
    p.add_argument('--dissim',     required=False, type=str,   default='')

    # Model hyperparameters.
    p.add_argument('--l1_coef',    required=False, type=float, default=0)
    p.add_argument('--lr',         required=False, type=float, default=0.001)

    # Job settings.
    p.add_argument('--use_cuda',   required=False, type=bool, default=True)
    p.add_argument('--n_workers',  required=False, type=int,  default=4)
    args = p.parse_args()

    # Put these values on `args` so their values are logged.
    args.cuda = args.use_cuda and torch.cuda.is_available()
    # Pinning memory only works when using GPUs.
    args.pin_memory = args.cuda

    main(args)
