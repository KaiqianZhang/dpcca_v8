"""=============================================================================
Autoencoder.
============================================================================="""

import torch
from   torch import nn
from   torch.nn import functional as F

# ------------------------------------------------------------------------------

class VAESigmoid(nn.Module):

    def __str__(self):
        return '<VAESigmoid>'

# ------------------------------------------------------------------------------

    def __init__(self, cfg, **kwargs):
        super(VAESigmoid, self).__init__()

        self.nc = cfg.N_CHANNELS
        self.w  = cfg.IMG_SIZE
        self.input_dim = self.nc * self.w * self.w

        mid_dim    = round(cfg.EMBEDDING_DIM / 2)
        latent_dim = cfg.EMBEDDING_DIM

        self.fc1   = nn.Linear(self.input_dim, mid_dim)
        self.mu    = nn.Linear(mid_dim,        latent_dim)
        self.sigma = nn.Linear(mid_dim,        latent_dim)

        self.fc3 = nn.Linear(latent_dim,     mid_dim)
        self.fc4 = nn.Linear(mid_dim,        self.input_dim)

# ------------------------------------------------------------------------------

    def encode(self, x):
        x = x.view(-1, self.input_dim)
        h = F.relu(self.fc1(x))
        return self.mu(h), self.sigma(h)

# ------------------------------------------------------------------------------

    def decode(self, z):
        h = F.relu(self.fc3(z))
        xr = torch.sigmoid(self.fc4(h))
        xr = xr.view(-1, self.nc, self.w, self.w)
        return xr

# ------------------------------------------------------------------------------

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x
