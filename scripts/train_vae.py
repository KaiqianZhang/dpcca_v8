"""=============================================================================
Train DMCM model.
============================================================================="""

import torch
from torch.autograd import Variable
from torch import nn, optim
from torch.nn import functional as F
from torchvision.utils import save_image

import cuda
from data.pacman.config import ToyConfig
from data import ToyDataset
from data import loader
from models import CVAE


# ------------------------------------------------------------------------------

def _init_weights(m):
    """Credit: https://discuss.pytorch.org/t/weight-initilzation/157/9.
    """
    if isinstance(m, nn.Conv2d):
        # Use Xavier normalization which helps with vanishing gradients:
        #
        #     http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
        #
        # We exclude bias nodes because they are constants and therefore have
        # zero variance.
        nn.init.xavier_uniform(m.weight.data)


# ------------------------------------------------------------------------------

def loss_function(x_recon, x, mu, logvar, n_pixels):
    bce_loss = F.binary_cross_entropy(x_recon,
                                      x.view(-1, n_pixels),
                                      size_average=False)

    # See Appendix B:
    #
    #     https://arxiv.org/abs/1312.6114
    #
    # Upshot:
    #
    #     0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    #
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce_loss + kld_loss


# ------------------------------------------------------------------------------

def train(model, train_loader, optimizer):
    model.train()
    train_loss = 0

    for x, _, _ in train_loader:
        optimizer.zero_grad()

        x = cuda.ize(Variable(x / 255.))
        _, x_recon, z_mu, z_logvar = model.forward(x)
        loss = loss_function(x_recon, x, z_mu, z_logvar, model.n_pixels)
        loss.backward()

        train_loss += loss.data.mean()
        optimizer.step()

    train_loss /= len(train_loader.dataset)
    return train_loss


# ------------------------------------------------------------------------------

def test(model, test_loader):
    model.eval()
    test_loss = 0
    for x, _, _ in test_loader:
        x = cuda.ize(Variable(x / 255.))
        _, recon_batch, mu, logvar = model.forward(x)
        loss = loss_function(recon_batch, x, mu, logvar, model.n_pixels)
        test_loss += loss.data.mean()
    test_loss /= len(test_loader.dataset)
    return test_loss


# ------------------------------------------------------------------------------

def sample_and_save(i):
    # `randn` returns a tensor filled with random numbers from a normal
    # distribution with 0 mean and variance of 1.
    sample = cuda.ize(Variable(torch.randn(N_TO_PRINT, 20)))
    sample = model.decode(sample).cpu()
    images = sample.data.view(64, 3, 32, 32)
    save_image(images, 'scratch/sample_%s.png' % i)


# ------------------------------------------------------------------------------

if __name__ == '__main__':
    BATCH_SIZE  = 128
    NUM_WORKERS = 4
    PIN_MEMORY  = True
    CV_PCT      = 0.2
    N_EPOCHS    = 1000
    N_TO_PRINT  = 64

    cuda.set_preference(True)

    config = ToyConfig()
    dataset = ToyDataset(config)

    model = CVAE(config)
    model = cuda.ize(model)
    model.apply(_init_weights)

    optimizer = optim.Adam(model.parameters(),
                           lr=1e-3)

    train_loader, test_loader = loader.get_data_loaders(config,
                                                        BATCH_SIZE,
                                                        NUM_WORKERS,
                                                        cuda.CUDA_IS_AVAILABLE,
                                                        CV_PCT)

    for epoch in range(0, N_EPOCHS):
        train_loss = train(model, train_loader, optimizer)
        test_loss  = test(model, test_loader)
        print(train_loss, test_loss)
        sample_and_save(epoch)
