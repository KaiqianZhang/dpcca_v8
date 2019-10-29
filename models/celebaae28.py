"""=============================================================================
Autoencoder based on "vanilla_ae" that has good results on CelebA:

Credit: https://github.com/bhpfelix/Variational-Autoencoder-PyTorch
============================================================================="""

import torch.nn as nn

# ------------------------------------------------------------------------------

class CelebAAE28(nn.Module):

    def __init__(self, cfg, **kwargs):
        super(CelebAAE28, self).__init__()

        nc  = cfg.N_CHANNELS
        ngf = cfg.IMG_SIZE
        ndf = cfg.IMG_SIZE
        latent_variable_size = cfg.EMBEDDING_DIM

        self.nc  = nc
        self.ngf = ngf
        self.ndf = ndf

        # encoder
        self.e1  = nn.Conv2d(nc, ndf, 4, 2, 1)
        self.bn1 = nn.BatchNorm2d(ndf)

        self.e2  = nn.Conv2d(ndf, ndf*2, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(ndf*2)

        self.e3  = nn.Conv2d(ndf*2, ndf*4, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(ndf*4)

        self.fc1 = nn.Linear(112*3*3, latent_variable_size)
        self.fc2 = nn.Linear(latent_variable_size, 112*3*3)

        self.up3 = nn.Upsample(scale_factor=2)
        self.pd3 = nn.ReplicationPad2d(1)
        self.d4  = nn.Conv2d(ngf*4, ngf*2, 3, 1)
        self.bn8 = nn.BatchNorm2d(ngf*2, 1.e-3)

        self.up4 = nn.Upsample(scale_factor=2)
        self.pd4 = nn.ReplicationPad2d(2)
        self.d5  = nn.Conv2d(ngf*2, ngf, 3, 1)
        self.bn9 = nn.BatchNorm2d(ngf, 1.e-3)

        self.up5 = nn.Upsample(scale_factor=2)
        self.pd5 = nn.ReplicationPad2d(1)
        self.d6  = nn.Conv2d(ngf, nc, 3, 1)

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu      = nn.ReLU()
        self.sigmoid   = nn.Sigmoid()

# ------------------------------------------------------------------------------

    def encode(self, x):
        h1 = self.leakyrelu(self.bn1(self.e1(x)))
        h2 = self.leakyrelu(self.bn2(self.e2(h1)))
        h3 = self.leakyrelu(self.bn3(self.e3(h2)))
        h3 = h3.view(-1, 112*3*3)
        return self.fc1(h3)

# ------------------------------------------------------------------------------

    def decode(self, z):
        h3 = self.relu(self.fc2(z))
        h3 = h3.view(-1, 112, 3, 3)
        h4 = self.leakyrelu(self.bn8(self.d4(self.pd3(self.up3(h3)))))
        h5 = self.leakyrelu(self.bn9(self.d5(self.pd4(self.up4(h4)))))
        xr = self.sigmoid(self.d6(self.pd5(self.up5(h5))))
        return xr

# ------------------------------------------------------------------------------

    def forward(self, x):
        z  = self.encode(x.view(-1, self.nc, self.ndf, self.ngf))
        xr = self.decode(z)
        return xr
