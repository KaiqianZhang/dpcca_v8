"""=============================================================================
Factor analysis.
============================================================================="""

import torch
import cuda
import linalg as LA
from   torch.distributions.multivariate_normal import MultivariateNormal as MVN

# ------------------------------------------------------------------------------

diag  = torch.diag
det   = torch.det
exp   = torch.exp
inv   = torch.inverse
log   = torch.log
outer = torch.ger
tr    = torch.trace

device = cuda.device()

# ------------------------------------------------------------------------------

class FA(object):

    def __init__(self, latent_dim, data_dim, n_iters=1):

        self.latent_dim = latent_dim
        self.p = data_dim
        self.n_iters = n_iters
        Lambda, Psi_diag = self.init_params()
        self.Lambda = Lambda
        self.Psi_diag = Psi_diag

# ------------------------------------------------------------------------------

    def fit(self, y):
        self.em(y)

# ------------------------------------------------------------------------------

    def em(self, y):
        Lambda, Psi_diag = self.Lambda, self.Psi_diag
        nlls = []

        for _ in range(self.n_iters):
            Lambda, Psi_diag = self.em_step(y, Lambda, Psi_diag)
            with torch.no_grad():
                nll = self.neg_log_likelihood(y, Lambda, Psi_diag)
                nlls.append(nll)

        self.Lambda = Lambda
        self.Psi_diag = Psi_diag
        self.nlls = nlls

# ------------------------------------------------------------------------------

    def em_step(self, y, Lambda, Psi_diag):
        k = self.latent_dim
        p, n = y.shape

        PLL_inv = LA.woodbury_inv(Psi_diag, Lambda, Lambda.t(), k)

        # E-step: compute expected moments for latent variable z.
        # -------------------------------------------------------
        Ez  = self.E_z_given_y(Lambda, PLL_inv, y)
        Ezz = self.E_zzT_given_y(Lambda, PLL_inv, y, k)

        # M-step: compute optimal Lambda and Psi.
        # ---------------------------------------

        # Compute Lambda_new (Equation 5, G&H 1996).
        Lambda_lterm = LA.sum_outers(y, Ez)
        Lambda_new   = Lambda_lterm @ inv(Ezz)

        # Compute Psi_diag_new (Equation 6, G&H 1996). Must use Lambda_new!
        Psi_rterm    = LA.sum_outers(y, y) - LA.sum_outers(Lambda_new @ Ez, y)
        Psi_diag_new = 1./n * diag(Psi_rterm)

        return Lambda_new, Psi_diag_new

# ------------------------------------------------------------------------------

    def E_z_given_y(self, L, PLL_inv, y):
        assert len(PLL_inv.shape) == 2
        beta = L.t() @ PLL_inv
        return beta @ y

# ------------------------------------------------------------------------------

    def E_zzT_given_y(self, L, PLL_inv, y, k):
        _, n = y.shape
        assert len(PLL_inv.shape) == 2
        beta = L.t() @ PLL_inv
        I    = torch.eye(k, device=device)
        bL   = beta @ L
        by   = beta @ y
        byyb = torch.einsum('ib,ob->io', [by, by])
        return n * (I - bL) + byyb

# ------------------------------------------------------------------------------

    def estimate_z_given_y(self, y):
        k = self.latent_dim
        Lambda, Psi_diag = self.Lambda, self.Psi_diag
        PLL_inv = LA.woodbury_inv(Psi_diag, Lambda, Lambda.t(), k)
        return self.E_z_given_y(Lambda, PLL_inv, y)

# ------------------------------------------------------------------------------

    def neg_log_likelihood(self, y, Lambda=None, Psi_diag=None):
        p, n = y.shape
        k = self.latent_dim

        if Lambda is None and Psi_diag is None:
            Lambda, Psi_diag = self.Lambda, self.Psi_diag

        PLL_inv = LA.woodbury_inv(Psi_diag, Lambda, Lambda.t(), k)
        Ez  = self.E_z_given_y(Lambda, PLL_inv, y).t()
        Ezz = self.E_zzT_given_y(Lambda, PLL_inv, y, k).t()

        inv_Psi = diag(LA.diag_inv(Psi_diag))

        A = 1/2. * diag(y.t() @ inv_Psi @ y)
        B = diag(y.t() @ inv_Psi @ Lambda @ Ez.t())
        C = 1/2. * tr(Lambda.t() @ inv_Psi @ Lambda @ Ezz)
        rterm_sum = (A - B).sum() + C

        logdet = -n/2. * log(det(diag(Psi_diag)))

        ll = (logdet - rterm_sum).item()
        nll = -ll
        return nll

# ------------------------------------------------------------------------------

    def init_params(self):
        p = self.p
        k  = self.latent_dim
        Lambda = torch.randn(p, k).to(device)
        Psi_diag = torch.ones(p).to(device)
        return Lambda, Psi_diag

    # ------------------------------------------------------------------------------

    def sample(self, y, n=None):
        k = self.latent_dim

        if n and n > y.shape[1]:
            raise AttributeError('More samples than estimated z variables.')
        elif not n:
            n = y.shape[1]

        PLL_inv = LA.woodbury_inv(self.Psi_diag, self.Lambda, self.Lambda.t(), k)
        z = self.E_z_given_y(self.Lambda, PLL_inv, y)

        m = self.Lambda @ z

        y = torch.empty(self.p, n)

        for i in range(n):
            y[:, i] = MVN(m[:, i], diag(self.Psi_diag)).sample()

        return y