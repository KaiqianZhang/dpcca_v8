"""=============================================================================
Deep CCA.
============================================================================="""

import torch
from   torch import nn

import cuda

# ------------------------------------------------------------------------------

diag  = torch.diag
inv   = torch.inverse
mm    = torch.matmul
sqrt  = torch.sqrt
trace = torch.trace

# ------------------------------------------------------------------------------

class DCCA(nn.Module):

    def __init__(self, cfg):
        super(DCCA, self).__init__()
        self.image_net = cfg.get_image_net()
        self.genes_net = cfg.get_genes_net()

# ------------------------------------------------------------------------------

    def forward(self, x):
        x1, x2 = x
        h1 = self.image_net.encode(x1)
        h2 = self.genes_net.encode(x2)
        return h1, h2

# ------------------------------------------------------------------------------

    def cca_loss(self, h1, h2):
        """Compute the negative correlation between the output of the two neural
        networks, h1 and h2. We don't actually need to analytically compute the
        derivative as described in the paper. We can just use automatic
        differentiation.
        """
        h1 = h1 - h1.mean(dim=0)
        h2 = h2 - h2.mean(dim=0)

        h1 = h1.t()
        h2 = h2.t()

        s11 = covariance(h1)
        s12 = covariance(h1, h2)
        s22 = covariance(h2)

        s11_root_inv = sqrt(inv(s11))
        s22_root_inv = sqrt(inv(s22))

        T = s11_root_inv @ s12 @ s22_root_inv
        corr = sqrt(trace(T.t() @ T))

        # Return negative correlation because our optimizer will *minimize*.
        return -1 * corr

# ------------------------------------------------------------------------------

def covariance(X, Y=None):
    eps = 1e-4
    p, n = X.shape
    normalizer = (1.0 / (n - 1))
    if Y is None:
        # This term is added according DCCA paper (above Equation 1).
        reg = eps * torch.eye(p, device=cuda.device())
        return normalizer * X @ X.t() + reg
    return normalizer * X @ Y.t()

# ------------------------------------------------------------------------------

def sqrt_inv(X):
    """Calculate the root inverse of covariance matrices by using
    eigendecomposition.
    """
    S, Q = torch.eig(X, eigenvectors=True)
    S = S[:, 0]
    X_root_inv = Q @ inv(sqrt(diag(S))) @ Q.t()
    return X_root_inv
