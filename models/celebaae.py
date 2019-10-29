"""=============================================================================
Autoencoder based on "vanilla_ae" that has good results on CelebA:

Credit: https://github.com/bhpfelix/Variational-Autoencoder-PyTorch
============================================================================="""

from   torch import nn
from   torch.nn import functional as F

# ------------------------------------------------------------------------------

class CelebAAE(nn.Module):

    def __init__(self, cfg):
        super(CelebAAE, self).__init__()

        nc  = cfg.N_CHANNELS

        # Relates to the depth of feature maps carried through the generator.
        ngf = cfg.IMG_SIZE
        # Sets the depth of feature maps propagated through the discriminator.
        ndf = cfg.IMG_SIZE

        self.nc  = nc
        self.ngf = ngf
        self.ndf = ndf

        # Encoder (discriminator).
        self.e1  = nn.Conv2d(nc, ndf, 4, 2, 1)
        self.bn1 = nn.BatchNorm2d(ndf)

        self.e2  = nn.Conv2d(ndf, ndf*2, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(ndf*2)

        self.e3  = nn.Conv2d(ndf*2, ndf*4, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(ndf*4)

        self.e4  = nn.Conv2d(ndf*4, ndf*8, 4, 2, 1)
        self.bn4 = nn.BatchNorm2d(ndf*8)

        self.e5  = nn.Conv2d(ndf*8, ndf*8, 4, 2, 1)
        self.bn5 = nn.BatchNorm2d(ndf*8)

        self.fc1 = nn.Linear(ndf*8*4*4, cfg.IMG_EMBED_DIM)

        # Decoder (generator).
        self.d1  = nn.Linear(cfg.IMG_EMBED_DIM, ngf*8*2*4*4)

        self.pd1 = nn.ReplicationPad2d(1)
        self.d2  = nn.Conv2d(ngf*8*2, ngf*8, 3, 1)
        self.bn6 = nn.BatchNorm2d(ngf*8, 1.e-3)

        self.pd2 = nn.ReplicationPad2d(1)
        self.d3  = nn.Conv2d(ngf*8, ngf*4, 3, 1)
        self.bn7 = nn.BatchNorm2d(ngf*4, 1.e-3)

        self.pd3 = nn.ReplicationPad2d(1)
        self.d4  = nn.Conv2d(ngf*4, ngf*2, 3, 1)
        self.bn8 = nn.BatchNorm2d(ngf*2, 1.e-3)

        self.pd4 = nn.ReplicationPad2d(1)
        self.d5  = nn.Conv2d(ngf*2, ngf, 3, 1)
        self.bn9 = nn.BatchNorm2d(ngf, 1.e-3)

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
        h4 = self.leakyrelu(self.bn4(self.e4(h3)))
        h5 = self.leakyrelu(self.bn5(self.e5(h4)))
        h5 = h5.view(-1, self.ndf*8*4*4)
        return self.fc1(h5)

# ------------------------------------------------------------------------------

    def decode(self, z):
        h1 = self.relu(self.d1(z))
        h1 = h1.view(-1, self.ngf * 8 * 2, 4, 4)
        t1 = F.interpolate(h1, scale_factor=2)
        h2 = self.leakyrelu(self.bn6(self.d2(self.pd1(t1))))
        t2 = F.interpolate(h2, scale_factor=2)
        h3 = self.leakyrelu(self.bn7(self.d3(self.pd2(t2))))
        t3 = F.interpolate(h3, scale_factor=2)
        h4 = self.leakyrelu(self.bn8(self.d4(self.pd3(t3))))
        t4 = F.interpolate(h4, scale_factor=2)
        h5 = self.leakyrelu(self.bn9(self.d5(self.pd4(t4))))
        t5 = F.interpolate(h5, scale_factor=2)
        return self.sigmoid(self.d6(self.pd5(t5)))

# ------------------------------------------------------------------------------

    def forward(self, x):
        z = self.encode(x.view(-1, self.nc, self.ndf, self.ngf))
        xr = self.decode(z)
        return xr
