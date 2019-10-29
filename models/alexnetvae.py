"""=============================================================================
AlexNet-based variational autoencoder.
============================================================================="""

import torch
from   torch import nn
import torch.nn.functional as F
from   torch.autograd import Variable

import cuda

# ------------------------------------------------------------------------------

class AlexNetVAE(nn.Module):

    def __str__(self):
        return '<AlexNetVAE>'

# ------------------------------------------------------------------------------

    def __init__(self, cfg, **kwargs):
        super(AlexNetVAE, self).__init__()

        if 'input_dim' in kwargs:
            self.n_pixels = kwargs['input_dim']
        else:
            self.n_pixels = cfg.N_CHANNELS * cfg.IMG_SIZE * cfg.IMG_SIZE

        nc = 3

        self.conv1    = nn.Conv2d(nc,  64,  kernel_size=11, padding=2, stride=4)
        self.conv2    = nn.Conv2d(64,  192, kernel_size=5,  padding=2)
        self.conv3    = nn.Conv2d(192, 384, kernel_size=3,  padding=1)
        self.conv4    = nn.Conv2d(384, 256, kernel_size=3,  padding=1)
        self.conv5    = nn.Conv2d(256, 256, kernel_size=3,  padding=1)

        self.mu_net  = nn.Linear(256 * 6 * 6, cfg.LATENT_DIM)
        self.var_net = nn.Linear(256 * 6 * 6, cfg.LATENT_DIM)
        self.unravel = nn.Linear(cfg.LATENT_DIM, 256 * 6 * 6)

        self.up12 = nn.Upsample([12, 12])
        self.up24 = nn.Upsample([24, 24])
        self.up55 = nn.Upsample([55, 55], mode='bilinear', align_corners=False)

        self.unconv5    = nn.ConvTranspose2d(256, 256, kernel_size=3, padding=1)
        self.unconv4    = nn.ConvTranspose2d(256, 384, kernel_size=3, padding=1)
        self.unconv3    = nn.ConvTranspose2d(384, 192, kernel_size=3, padding=1)
        self.unconv2    = nn.ConvTranspose2d(192, 64,  kernel_size=5, padding=2)
        self.unconv1    = nn.ConvTranspose2d(64,  nc,  kernel_size=11,
                                             padding=2, stride=4,
                                             output_padding=1)

# ------------------------------------------------------------------------------

    def encode(self, x):
        # Input:  3,  224, 224

        # Output: 64, 55,  55
        x1e = F.relu(self.conv1(x))
        # Output: 64, 27, 27
        x1p = F.max_pool2d(x1e, kernel_size=3, stride=2)

        # Output: 192, 27, 27
        x2e = F.relu(self.conv2(x1p))
        # Output: 192, 13, 13
        x2p = F.max_pool2d(x2e, kernel_size=3, stride=2)

        # Output: 384, 13, 13
        x3e = F.relu(self.conv3(x2p))

        # Output: 256, 13, 13
        x4e = F.relu(self.conv4(x3e))

        # Output: 256, 13, 13
        x5e = F.relu(self.conv5(x4e))
        x5p = F.max_pool2d(x5e, kernel_size=3, stride=2)

        x6 = x5p.view(-1, 256 * 6 * 6)
        return self.mu_net(x6), self.var_net(x6)

# ------------------------------------------------------------------------------

    def reparameterize(self, mu, logvar):
        """Perform the reparameterization trick, i.e. sample z. See Figure 4 in:

            https://arxiv.org/pdf/1606.05908.pdf
        """
        if self.training:
            std = logvar.mul(0.5).exp_()
            # In a multi-GPU setting, ensure both tensors are on the same GPU so
            # we can multiply them.
            eps = cuda.ize(Variable(torch.Tensor(std.size()).normal_()),
                           device=std.device)
            return eps.mul(std).add_(mu)
        else:
            return mu

# ------------------------------------------------------------------------------

    def decode(self, z):
        z = self.unravel(z)
        z = z.view(-1, 256, 6, 6)

        # Input:  256, 6, 6
        # Output: 256, 12, 12
        x5d = self.up12(z)

        # Output: 256, 12, 12
        x4d = F.relu(self.unconv5(x5d))

        # Output: 384, 12, 12
        x3d = F.relu(self.unconv4(x4d))

        # Output: 192, 12, 12
        x2u = F.relu(self.unconv3(x3d))

        # Output: 192, 24, 24
        x2d = self.up24(x2u)

        # Output: 64, 24, 24
        x1u = F.relu(self.unconv2(x2d))

        # Output: 64, 55, 55
        x1d = self.up55(x1u)

        # Output: 3, 224, 224
        x_rec = self.unconv1(x1d)

        return x_rec

# ------------------------------------------------------------------------------

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return z, x_recon, mu, logvar

# ------------------------------------------------------------------------------

    def loss(self, recon_x, x, mu, logvar):
        """Compute the VAE loss.
        """
        recon = F.mse_loss(recon_x.view(-1, self.n_pixels),
                           x.view(-1, self.n_pixels),
                           size_average=False)
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # Return the loss as two numbers so we can log both.
        return recon, kld
