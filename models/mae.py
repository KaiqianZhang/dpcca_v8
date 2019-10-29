"""=============================================================================
Multimodal autoencoder.
============================================================================="""

import torch
from   torch import nn
from   torch.nn import functional as F

# ------------------------------------------------------------------------------

class MAE(nn.Module):

    def __init__(self, cfg):
        super(MAE, self).__init__()
        self.cfg = cfg
        self.image_net = cfg.get_image_net()
        self.genes_net = cfg.get_genes_net()

        self.enc = nn.Linear(cfg.IMG_EMBED_DIM + cfg.GENE_EMBED_DIM, 2)
        self.dec = nn.Linear(2, cfg.IMG_EMBED_DIM + cfg.GENE_EMBED_DIM)

# ------------------------------------------------------------------------------

    def forward(self, x):
        x1, x2 = x

        y1 = self.image_net.encode(x1)
        y2 = self.genes_net.encode(x2)

        y  = torch.cat([y1, y2], dim=1)
        z  = self.enc(y)
        yr = self.dec(z)
        y1r = yr[:, self.cfg.IMG_EMBED_DIM:]
        y2r = yr[:, :self.cfg.IMG_EMBED_DIM]

        x1r = self.image_net.decode(y1r)
        x2r = self.genes_net.decode(y2r)

        return x1r, x2r