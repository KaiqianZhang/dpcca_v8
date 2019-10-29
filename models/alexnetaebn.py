"""=============================================================================
AlexNet-based autoencoder with batch normalization. For more:

    Batch Normalization: Accelerating Deep Network Training by Reducing Internal
    Covariate Shift: https://arxiv.org/abs/1502.03167

Implementation: between fully connected layers and nonlinearities, meaning: not
after the final output layer:

    https://stats.stackexchange.com/a/302061/77289
============================================================================="""

from   torch import nn
import torch.nn.functional as F
from   torchvision import models

# ------------------------------------------------------------------------------

class AlexNetAEBN(nn.Module):

    def __init__(self, cfg):
        super(AlexNetAEBN, self).__init__()
        nc = cfg.N_CHANNELS

        self.n_pixels = nc * cfg.IMG_SIZE * cfg.IMG_SIZE
        self.alexnet  = models.alexnet(pretrained=True)
        self.alexnet.classifier[6] = nn.Linear(4096, cfg.IMG_EMBED_DIM)

        self.decoder = nn.Sequential(
            nn.Dropout(),
            nn.Linear(cfg.IMG_EMBED_DIM, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 256 * 6 * 6)
        )

        self.unconv5    = nn.ConvTranspose2d(256, 256, kernel_size=2, padding=0,
                                             stride=1)
        self.unconv5_bn = nn.BatchNorm2d(256)

        self.unconv4    = nn.ConvTranspose2d(256, 384, kernel_size=2, padding=0,
                                             stride=2)
        self.unconv4_bn = nn.BatchNorm2d(384)

        self.unconv3    = nn.ConvTranspose2d(384, 192, kernel_size=2, padding=0,
                                             stride=2)
        self.unconv3_bn = nn.BatchNorm2d(192)

        self.unconv2    = nn.ConvTranspose2d(192, 64,  kernel_size=4, padding=2,
                                             stride=2)
        self.unconv2_bn = nn.BatchNorm2d(64)

        # No batch normalization on the output layer.
        self.unconv1    = nn.ConvTranspose2d(64,  nc,  kernel_size=10,
                                             padding=0,stride=4,
                                             output_padding=2)

# ------------------------------------------------------------------------------

    def forward(self, x):
        z  = self.encode(x)
        xr = self.decode(z)
        return xr

# ------------------------------------------------------------------------------

    def encode(self, x):
        return self.alexnet.forward(x)

# ------------------------------------------------------------------------------

    def decode(self, z):
        z = self.decoder(z)
        z = z.view(-1, 256, 6, 6)

        # Input:  256, 6,  6
        # Output: 256, 7, 7
        x4d = F.relu(self.unconv5_bn(self.unconv5(z)))

        # Output: 384, 14, 14
        x3d = F.relu(self.unconv4_bn(self.unconv4(x4d)))

        # Output: 192, 28, 28
        x2d = F.relu(self.unconv3_bn(self.unconv3(x3d)))

        # Output: 64, 54, 54
        x1d = F.relu(self.unconv2_bn(self.unconv2(x2d)))

        # Output: 3, 224, 224
        xr  = self.unconv1(x1d)

        return xr

