"""=============================================================================
Autoencoder.
============================================================================="""

import numpy as np
from   torch import nn

# ------------------------------------------------------------------------------

class CrossModalityAE(nn.Module):

    def __name__(self):
        return 'CrossModalityAE'

# ------------------------------------------------------------------------------

    def __str__(self):
        return '<CrossModalityAE>'

# ------------------------------------------------------------------------------

    def __init__(self, cfg):
        super(CrossModalityAE, self).__init__()
        self.genes_net  = cfg.get_genes_net()
        cfg.N_GENES = cfg.N_PIXELS
        self.image_net  = cfg.get_image_net()

        self.nc = cfg.N_CHANNELS
        self.w  = cfg.IMG_SIZE

# ------------------------------------------------------------------------------

    def forward(self, x):
        return self.forward_double_cross(x)

# ------------------------------------------------------------------------------

    def forward_single_cross(self, x):
        x1, x2 = x
        z2 = self.genes_net.encode(x2)
        x1_from_z2 = self.image_net.decode(z2)
        return x1_from_z2

# ------------------------------------------------------------------------------

    def forward_double_cross(self, x):
        x1, x2 = x
        # x1 = x1.view(-1, np.prod(x1.shape[1:]))

        z1 = self.image_net.encode(x1)
        z2 = self.genes_net.encode(x2)

        x1_from_z1 = self.image_net.decode(z1)
        x1_from_z2 = self.image_net.decode(z2)
        x2_from_z2 = self.genes_net.decode(z2)
        x2_from_z1 = self.genes_net.decode(z1)

        # p = x1_from_z1.shape[0]
        # x1_from_z1 = x1_from_z1.view(p, self.nc, self.w, self.w)
        # x1_from_z2 = x1_from_z2.view(p, self.nc, self.w, self.w)

        return z1, z2, x1_from_z1, x1_from_z2, x2_from_z2, x2_from_z1
