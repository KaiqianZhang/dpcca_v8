"""=============================================================================
Linear model.
============================================================================="""

import torch.nn as nn

# ------------------------------------------------------------------------------

class Linear(nn.Module):

    def __str__(self):
        return '<Linear>'

    def __init__(self, cfg):
        """Initialize simple linear model.
        """
        super(Linear, self).__init__()
        self.ll = nn.Linear(cfg.N_GENES, cfg.LATENT_DIM)

# ------------------------------------------------------------------------------

    def forward(self, x):
        """Perform forward pass on neural network.
        """
        return self.ll(x)
