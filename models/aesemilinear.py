"""=============================================================================
Linear model.
============================================================================="""

from   torch import nn

# ------------------------------------------------------------------------------

class AESemiLinear(nn.Module):

    def __str__(self):
        return '<AESemiLinear>'

# ------------------------------------------------------------------------------

    def __init__(self, cfg):
        """Initialize simple linear model.
        """
        super(AESemiLinear, self).__init__()
        self.input_dim = cfg.N_GENES
        emb_dim  = cfg.GENE_EMBED_DIM

        # Use sigmoid because gene expression data is in [0, 1] range.
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 4096),
            nn.ReLU(),
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, emb_dim),
            nn.Sigmoid()
        )
        self.linear_layer = nn.Linear(emb_dim, self.input_dim)

# ------------------------------------------------------------------------------

    def encode(self, x):
        return self.encoder(x)

# ------------------------------------------------------------------------------

    def decode(self, z):
        return self.linear_layer(z)

# ------------------------------------------------------------------------------

    def forward(self, x):
        z = self.encode(x.view(-1, self.input_dim))
        return self.decode(z)
