"""=============================================================================
Deep variational canonical correlation analysis.
============================================================================="""

import torch
from   torch import nn
import cuda

# ------------------------------------------------------------------------------

class DVCCA(nn.Module):

    def __init__(self, cfg):
        super(DVCCA, self).__init__()
        self.image_net = cfg.get_image_net()
        self.genes_net = cfg.get_genes_net()
        self.latent_dim = cfg.EMBEDDING_DIM

# ------------------------------------------------------------------------------

    def forward(self, x):
        x1, x2 = x
        mu, log_var = self.image_net.encode(x1)
        z1 = self.reparameterize(mu, log_var)
        z2 = self.reparameterize(mu, log_var)
        x1r = self.image_net.decode(z1.unsqueeze(-1).unsqueeze(-1))
        x2r = self.genes_net.decode(z2)
        return x1r, x2r, mu, log_var

# ------------------------------------------------------------------------------

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

# ------------------------------------------------------------------------------

    def sample(self, _, n_samples):
        zs = torch.randn(n_samples, self.latent_dim).to(cuda.device())
        x1s = self.image_net.decode(zs.unsqueeze(-1).unsqueeze(-1)).cpu()
        x2s = self.genes_net.decode(zs).cpu()
        return x1s, x2s
