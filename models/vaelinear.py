"""=============================================================================
Linear variational autoencoder model.
============================================================================="""

from   torch import nn

# ------------------------------------------------------------------------------

class VAELinear(nn.Module):

    def __str__(self):
        return '<VAELinear>'

# ------------------------------------------------------------------------------

    def __init__(self, cfg):
        """Initialize simple linear model.
        """
        super(VAELinear, self).__init__()
        emb_dim        = cfg.EMBEDDING_DIM
        self.input_dim = cfg.N_GENES
        self.fc1       = nn.Linear(self.input_dim, 2*emb_dim)
        self.mu        = nn.Linear(2*emb_dim, emb_dim)
        self.sigma     = nn.Linear(2*emb_dim, emb_dim)
        self.fc2       = nn.Linear(emb_dim, self.input_dim)

# ------------------------------------------------------------------------------

    def encode(self, x):
        z = self.fc1(x)
        return self.mu(z), self.sigma(z)

# ------------------------------------------------------------------------------

    def decode(self, z):
        return self.fc2(z)

# ------------------------------------------------------------------------------

    def forward(self, x):
        z = self.encode(x.view(-1, self.input_dim))
        return self.decode(z)
