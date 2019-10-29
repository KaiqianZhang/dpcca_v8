"""=============================================================================
Nonlinear encoder, linear decoder.
============================================================================="""

from   torch import nn
from   torch.nn import functional as F

# ------------------------------------------------------------------------------

class GeneAE(nn.Module):

    def __str__(self):
        return '<GeneAE>'

# ------------------------------------------------------------------------------

    def __init__(self, cfg, **kwargs):
        """Initialize simple linear model.
        """
        super(GeneAE, self).__init__()
        if 'input_dim' in kwargs:
            self.input_dim = kwargs['input_dim']
        else:
            self.input_dim = cfg.N_GENES
        emb_dim  = cfg.EMBEDDING_DIM
        self.fc1 = nn.Linear(self.input_dim, 4096)
        self.fc2 = nn.Linear(4096, emb_dim)
        # This name is useful for readability in other areas of the codebase.
        self.linear_decoder = nn.Linear(emb_dim, self.input_dim)

# ------------------------------------------------------------------------------

    def encode(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# ------------------------------------------------------------------------------

    def decode(self, z):
        return self.linear_decoder(z)

# ------------------------------------------------------------------------------

    def forward(self, x):
        z = self.encode(x.view(-1, self.input_dim))
        return self.decode(z)
