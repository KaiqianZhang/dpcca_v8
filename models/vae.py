"""=============================================================================
Variational autoencoder.
============================================================================="""

import torch
from   torch import nn
from   torch.nn import functional as F

# ------------------------------------------------------------------------------

class VAE(nn.Module):

    def __str__(self):
        return '<VAE>'

    def __init__(self, cfg, **kwargs):
        super(VAE, self).__init__()

        self.input_dim = cfg.N_CHANNELS * cfg.IMG_SIZE * cfg.IMG_SIZE
        mid_dim        = cfg.VAE_MID_DIM
        if 'pcca_z_dim' in kwargs:
            latent_dim = kwargs['pcca_z_dim']
        else:
            latent_dim = cfg.EMBEDDING_DIM

        self.fc1  = nn.Linear(self.input_dim, mid_dim)
        self.fc21 = nn.Linear(mid_dim,        latent_dim)
        self.fc22 = nn.Linear(mid_dim,        latent_dim)
        self.fc3  = nn.Linear(latent_dim,     mid_dim)
        self.fc4  = nn.Linear(mid_dim,        self.input_dim)

# ------------------------------------------------------------------------------

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

# ------------------------------------------------------------------------------

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

# ------------------------------------------------------------------------------

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return F.sigmoid(self.fc4(h3))

# ------------------------------------------------------------------------------

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_dim))
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar, z

# ------------------------------------------------------------------------------

    def loss(self, recon_x, x, mu, logvar):
        """Compute the VAE loss.
        """
        n_pixels = self.input_dim
        recon = F.mse_loss(recon_x, x.view(-1, n_pixels), size_average=False)
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon + kld
