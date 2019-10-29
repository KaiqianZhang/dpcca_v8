"""=============================================================================
DCGAN-based autoencoder with a 128x128 input. See:

    https://github.com/pytorch/examples/issues/70
============================================================================="""

import torch
from   torch import nn
from   torch.nn import functional as F

# ------------------------------------------------------------------------------

class DCGANVAE128(nn.Module):

    def __init__(self, cfg):
        super(DCGANVAE128, self).__init__()

        self.n_pixels = cfg.N_PIXELS
        nc = cfg.N_CHANNELS

        # Relates to the depth of feature maps carried through the generator.
        ndf = 64
        # Sets the depth of feature maps propagated through the discriminator.
        ngf = 64

        self.discriminator = nn.Sequential(
            # input is (nc) x 128 x 128
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 64 x 64
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 32 x 32
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 16 x 16
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 8 x 8
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*16) x 4 x 4
            nn.Conv2d(ndf * 16, cfg.IMG_EMBED_DIM, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # state size. cfg.IMG_EMBED_DIM
        )

        self.mu_net     = nn.Linear(cfg.IMG_EMBED_DIM, cfg.IMG_EMBED_DIM)
        self.logvar_net = nn.Linear(cfg.IMG_EMBED_DIM, cfg.IMG_EMBED_DIM)

        self.generator = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(cfg.IMG_EMBED_DIM, ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            # state size. (ngf*16) x 4 x 4
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 8 x 8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 16 x 16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 32 x 32
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 64 x 64
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 128 x 128
        )

# ------------------------------------------------------------------------------

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        xr = self.decode(z)
        return xr, mu, logvar

# ------------------------------------------------------------------------------

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

# ------------------------------------------------------------------------------

    def encode(self, x):
        z = self.discriminator(x)
        z = z.view(x.shape[0], -1)
        return self.mu_net(z), self.logvar_net(z)

# ------------------------------------------------------------------------------

    def decode(self, z):
        z = z.unsqueeze(-1).unsqueeze(-1)
        return self.generator(z)

# ------------------------------------------------------------------------------

    def loss(self, recon_x, x, mu, logvar):
        recon = F.mse_loss(recon_x, x, size_average=False)
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon, kld
