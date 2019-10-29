"""=============================================================================
Convolutional autoencder.

Based on the following for reproducibility and baseline comparisons:

    https://github.com/JordanAsh/CAE/blob/master/cae.py
============================================================================="""

from torch import nn

# ------------------------------------------------------------------------------

class CAE(nn.Module):

    def __str__(self):
        return '<CAE>'

# ------------------------------------------------------------------------------

    def __name__(self):
        return 'CAE'

# ------------------------------------------------------------------------------

    def __init__(self, cfg, **kwargs):
        super(CAE, self).__init__()

        assert cfg.IMG_SIZE == 128

        nc     = cfg.N_CHANNELS
        ks_enc = 4  # Kernel size for encoder
        ks_dec = 5  # Kernel size for decoder
        p_size = 2  # Pooling size
        stride = 1
        pad    = 2  # Padding

        self.encoder = nn.Sequential(
            # Input:  3, 128, 128
            # Output: 8, 64,  64
            nn.Conv2d(nc, 8, ks_enc, stride=stride, padding=pad),
            nn.MaxPool2d(p_size),
            nn.ReLU(True),

            # Input:  8,  64,  64
            # Output: 16, 32,  32
            nn.Conv2d(8, 16, ks_enc, stride=stride, padding=pad),
            nn.MaxPool2d(p_size),
            nn.ReLU(True),

            # Input:  16, 32, 32
            # Output: 32, 16, 16
            nn.Conv2d(16, 32, ks_enc, stride=stride, padding=pad),
            nn.MaxPool2d(p_size),
            nn.ReLU(True),

            # Input:  16, 16, 16
            # Output: 64, 8,  8
            nn.Conv2d(32, 64, ks_enc, stride=stride, padding=pad),
            nn.MaxPool2d(p_size),
            nn.ReLU(True),

            # Input:  64,  8, 8
            # Output: 128, 4, 4
            nn.Conv2d(64, 128, ks_enc, stride=stride, padding=pad),
            nn.MaxPool2d(p_size),
            nn.ReLU(True)
        )

        # We separate this into two models in order to
        self.m1 = nn.Sequential(
            nn.Linear(128*4*4, cfg.LATENT_DIM),
            nn.ReLU(True)
        )
        self.m2 = nn.Sequential(
            nn.Linear(cfg.LATENT_DIM, 128*4*4),
            nn.ReLU(True)
        )

        self.decoder = nn.Sequential(
            # Input:  128, 4, 4
            # Output: 64,  8, 8
            nn.Upsample(scale_factor=p_size),
            nn.Conv2d(128, 64, ks_dec, stride=1, padding=pad),
            nn.ReLU(True),

            # Input:  64, 8,  8
            # Output: 32, 16, 16
            nn.Upsample(scale_factor=p_size),
            nn.Conv2d(64, 32, ks_dec, stride=stride, padding=pad),
            nn.ReLU(True),

            # Input:  32, 16, 16
            # Output: 16, 32, 32
            nn.Upsample(scale_factor=p_size),
            nn.Conv2d(32, 16, ks_dec, stride=stride, padding=pad),
            nn.ReLU(True),

            # Input:  16, 32, 32
            # Output: 8,  64, 64
            nn.Upsample(scale_factor=p_size),
            nn.Conv2d(16, 8, ks_dec, stride=stride, padding=pad),
            nn.ReLU(True),

            # Input:  8, 64,  64
            # Output: 3, 128, 128
            nn.Upsample(scale_factor=p_size),
            nn.Conv2d(8, nc, ks_dec, stride=stride, padding=pad),
            nn.Sigmoid()
        )

# ------------------------------------------------------------------------------

    def forward(self, x):
        """Return reconstructed x after encoding and decoding.
        """
        x = self.encoder(x)
        x = x.view(-1, 128*4*4)
        z = self.m1(x)
        x = self.m2(z)
        x = x.view(x.size(0), 128, 4, 4)
        x_recon = self.decoder(x)
        return x_recon
