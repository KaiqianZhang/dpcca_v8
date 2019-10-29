"""=============================================================================
Autoencoder.
============================================================================="""

import numpy as np
from   torch import nn

# ------------------------------------------------------------------------------

class VAETanH(nn.Module):

    def __str__(self):
        return '<VAETanH>'

# ------------------------------------------------------------------------------

    def __init__(self, cfg, **kwargs):
        super(VAETanH, self).__init__()

        assert cfg.EMBEDDING_DIM < 12
        self.nc = cfg.N_CHANNELS
        self.w  = cfg.IMG_SIZE
        self.input_dim = cfg.N_GENES

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 2*cfg.EMBEDDING_DIM)
        )
        self.mu    = nn.Linear(2*cfg.EMBEDDING_DIM, cfg.EMBEDDING_DIM)
        self.sigma = nn.Linear(2*cfg.EMBEDDING_DIM, cfg.EMBEDDING_DIM)

        self.decoder = nn.Sequential(
            nn.Linear(cfg.EMBEDDING_DIM, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, self.input_dim)
        )

# ------------------------------------------------------------------------------

    def encode(self, x):
        x = x.view(-1, np.prod(x.shape[1:]))
        h = self.encoder(x)
        return self.mu(h), self.sigma(h)

# ------------------------------------------------------------------------------

    def decode(self, z):
        x = self.decoder(z)
        # return x.view(-1, self.nc * self.w * self.w)
        return x.view(-1, self.input_dim)

# ------------------------------------------------------------------------------

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x
