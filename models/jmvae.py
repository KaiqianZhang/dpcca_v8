"""=============================================================================
Joint multimodal variational autoencoder. See Suzuki 2017:

    https://openreview.net/pdf?id=BkL7bONFe
============================================================================="""

import torch
from   torch import nn
from   torch.nn import functional as F

# ------------------------------------------------------------------------------

class JMVAE(nn.Module):

    def __str__(self):
        return '<JMVAE>'

    def __init__(self, cfg):
        super(JMVAE, self).__init__()
        self.image_net = cfg.get_image_net()
        self.genes_net = cfg.get_genes_net()

        self.mu_concat     = nn.Linear(cfg.LATENT_DIM * 2, cfg.LATENT_DIM)
        self.logvar_concat = nn.Linear(cfg.LATENT_DIM * 2, cfg.LATENT_DIM)

# ------------------------------------------------------------------------------

    def forward(self, x):
        x1, x2 = x

        # Encode each data point and combine it into a single embedding.
        x1_mu, x1_logvar = self.image_net.encode(x1)
        x2_mu, x2_logvar = self.genes_net.encode(x2)

        # Concatenate mu and var into (2 * latent_dim)-dimensional vectors, then
        # learned linear combination to produce (latent_dim)-dimensional vector.
        mu_cat     = torch.cat((x1_mu, x2_mu), dim=1)
        mu         = self.mu_concat(mu_cat)
        logvar_cat = torch.cat((x1_logvar, x2_logvar), dim=1)
        logvar     = self.logvar_concat(logvar_cat)

        z = self.reparameterize(mu, logvar)

        x1_recon = self.image_net.decode(z)
        x2_recon = self.genes_net.decode(z)

        return x1_recon, x2_recon, mu, logvar

# ------------------------------------------------------------------------------

    def reparameterize(self, mu, logvar):
        """The reparameterization trick. See Kingma 2014.
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

# ------------------------------------------------------------------------------

    def loss(self, x1, x1_recon, x2, x2_recon, mu, logvar):
        """Compute the VAE loss.
        """
        recon1 = F.mse_loss(x1_recon, x1, size_average=False)
        recon2 = F.mse_loss(x2_recon, x2, size_average=False)
        kld    = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon1, recon2, kld
