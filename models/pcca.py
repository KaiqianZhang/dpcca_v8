"""=============================================================================
Probabilistic canonical correlation analysis. For references in comments:

    A Probabilistic Interpretation of Canonical Correlation Analysis.
    Bach, Jordan (2006).

    The EM algorithm for mixtures of factor analyzers.
    Ghahramani, Hinton (1996).

    Machine Learning: A Probabilistic Perspective
    Murphy (2006)

    Bayesian Reasoning and Machine Learning
    David Barber (2012)

For derivations in detail:

    Factor Analysis in Detail
    Gundersen (2018)
    http://gregorygundersen.com/blog/2018/08/08/factor-analysis/

    Probabilistic Canonical Correlation Analysis in Detail
    Gundersen (2018)
    http://gregorygundersen.com/blog/2018/09/10/pcca/

============================================================================="""

import numpy as np
import random
import torch
from   torch import nn
from   torch.distributions.multivariate_normal import MultivariateNormal as MVN

import cuda
import linalg as LA

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

class PCCA(nn.Module):

    def __init__(self, latent_dim, dims, gene_low_rank, max_iters=100,
                 private_z=True, differentiable=True, use_svd=False,
                 debug=False):
        """Initialize the probabilistic CCA model.

        :param latent_dim: The latent variable's dimension.
        :param dims:       Each modality's dimension.
        :param max_iters:  The number of EM iterations.
        """
        super(PCCA, self).__init__()

        self.latent_dim    = latent_dim
        self.max_iters     = max_iters
        self.use_svd       = use_svd
        self.gene_low_rank = gene_low_rank
        self.debug         = debug

        # For a description of PCCA with modality-specific private latent
        # variables, see Murphy (2006), p. 409.
        self.private_z = private_z

        # This controls whether or not we backprop through the PCCA parameters.
        self.differentiable = differentiable

        p1, p2  = dims
        self.p1 = p1
        self.p2 = p2

        Lambda1, Lambda2, B1, B2, log_Psi1_diag, log_Psi2_diag \
            = self.init_params()

        if self.differentiable:
            self.Lambda1 = nn.Parameter(Lambda1)
            self.Lambda2 = nn.Parameter(Lambda2)
            self.log_Psi1_diag = nn.Parameter(log_Psi1_diag)
            self.log_Psi2_diag = nn.Parameter(log_Psi2_diag)
            if self.private_z:
                self.B1 = nn.Parameter(B1)
                self.B2 = nn.Parameter(B2)
        else:
            self.Lambda1 = Lambda1
            self.Lambda2 = Lambda2
            self.log_Psi1_diag = log_Psi1_diag
            self.log_Psi2_diag = log_Psi2_diag
            if self.private_z:
                self.B1 = B1
                self.B2 = B2

# ------------------------------------------------------------------------------

    def __repr__(self):
        """Compute a printable representation of a PCCA object.

        :return: Human-readable string of the model.
        """
        msg = '\n  properties:\n'
        for k, v in self.__dict__.items():
            if k.startswith('_'):
                continue
            msg += '    %s: %s\n' % (k, v)
        msg += '  parameters:\n'
        params = ['Lambda1', 'Lambda2', 'log_Psi1_diag', 'log_Psi2_diag']
        if self.private_z:
            params[2:2] = ['B1', 'B2']
        msg += '\n'.join(['    (%s) %s' % (i, p) for i, p in enumerate(params)])
        return msg

# ------------------------------------------------------------------------------

    def forward(self, y):
        """Fit the probabilistic CCA model using Expectation-Maximization.

        :param y: Observations of shape (n_features, n_samples).
        """
        k = 3 * self.latent_dim if self.private_z else self.latent_dim

        if self.use_svd:
            Lambda, Psi_diag = self.em_svd(y)
            # z = self.transform(y)
        else:
            Lambda, Psi_diag = self.em_bishop(y)
        PLL_inv = LA.woodbury_inv(Psi_diag, Lambda, Lambda.t(), k)
        z = self.E_z_given_y(Lambda, PLL_inv, y)

        yr = self.reparameterize(Lambda, Psi_diag, z)
        y1 = yr[:self.p1]
        y2 = yr[self.p1:]

        if not self.differentiable:
            self.save_new_params(Lambda, Psi_diag)

        return y1.t(), y2.t()

# ------------------------------------------------------------------------------

    def em_svd(self, y, tol=1e-2):
        """Perform Expectation-Maximization using SVD. This implementation is
        based on Algorithm 21.1 in "Bayesian Reasoning and Machine Learning"
        by David Barber:

            http://www.cs.ucl.ac.uk/staff/d.barber/brml

        :param y: Observations of shape (n_features, n_samples).
        :return:  Parameters that maximize the likelihood given the data.
        """
        # TODO: Simplify API to handle y.shape == (n_samples, n_features).
        y     = y.t()

        k = 3 * self.latent_dim if self.private_z else self.latent_dim
        SMALL = torch.tensor(1e-12, device=device)
        Lambda, Psi_diag = self.tile_params()
        n, p  = y.shape

        # Fascinating. This constant comes from sklearn. I have no idea where it
        # comes from, but my guess is that it scales the log likelihood to be
        # consistently sized somehow.
        llconst = torch.tensor(p * np.log(2. * np.pi) + k, device=device)
        nsqrt   = torch.tensor(np.sqrt(n), device=device)
        var     = y.var(dim=0)

        log_likes = []
        old_ll    = -float('inf')
        iters_req = self.max_iters

        for i in range(self.max_iters):

            sqrt_Psi_diag = Psi_diag.sqrt()
            y_tilde       = y / (sqrt_Psi_diag * nsqrt)

            U, S, Vt = torch.svd(y_tilde.t())
            U = U[:, :k]
            S = S[:k]
            S = S**2

            # Subtract identity matrix and take max for numerical stability.
            zeros    = torch.zeros(1, device=device)

            # Equivalent to `sqrt(S - Ik)` with numerical stability.
            tmp      = torch.max(S - 1, zeros).sqrt()

            # Equivalent to: `Lambda = diag(sqrt_Psi_diag) @ U @ tmp`.
            Lambda   = ((U * tmp).t() * sqrt_Psi_diag).t()

            # Equivalent to: `diag(torch.max(var - Lambda @ Lambda.t(), SMALL))`
            Psi_diag = torch.max(var - (Lambda**2).sum(dim=1), SMALL)

            # Compute log likelihood if used in convergence analysis.
            if self.max_iters > 1:
                unexp_var = S.norm(2).sqrt()
                ll  = llconst + S.log().sum()
                ll += unexp_var + Psi_diag.log().sum()
                ll *= -n / 2.
                log_likes.append(ll.item())
                if torch.abs(ll - old_ll) < tol:
                    iters_req = i+1
                    break
                old_ll = ll

        self.nlls      = log_likes
        self.iters_req = iters_req

        return Lambda, Psi_diag

# ------------------------------------------------------------------------------

    def transform(self, y):
        """Apply dimensionality reduction to y using fitted parameters.

        See Bishop 12.66 or Barber 21.2.33.
        """
        # TODO: Simplify API to handle y.shape == (n_samples, n_features).
        y = y.t()
        k = 3 * self.latent_dim if self.private_z else self.latent_dim
        Lambda, Psi_diag = self.tile_params()

        Ik = torch.eye(k, device=device)

        # Equivalent to: `Lambda.t() @ inv(diag(Psi_diag))`.
        LP_inv = Lambda.t() / Psi_diag
        G = inv(Ik + LP_inv @ Lambda)

        z = G @ LP_inv @ y.t()
        return z

# ------------------------------------------------------------------------------

    def em_bishop(self, y):
        """Perform Expectation-Maximization.

        :param y: Observations of shape (n_features, n_samples).
        :return:  Parameters that maximize the likelihood given the data.
        """
        Lambda, Psi_diag = self.tile_params()
        nlls = []

        for _ in range(self.max_iters):
            Lambda, Psi_diag = self.em_bishop_step(y, Lambda, Psi_diag)
            if self.debug:
                nll = self.neg_log_likelihood(y, Lambda, Psi_diag)
                nlls.append(nll)

        self.nlls      = nlls
        self.iters_req = self.max_iters
        return Lambda, Psi_diag

# ------------------------------------------------------------------------------

    def em_bishop_step(self, y, Lambda, Psi_diag):
        """Equations 5 and 6 in Ghahramani and Hinton (1996).

        :param y:        Observations of shape (n_features, n_samples).
        :param Lambda:   Current Lambda.
        :param Psi_diag: Current Psi_diag.
        :return:         Maximum likelihood parameters for the current step.
        """
        k = 3 * self.latent_dim if self.private_z else self.latent_dim
        p, n = y.shape

        PLL_inv = LA.woodbury_inv(Psi_diag, Lambda, Lambda.t(), k)

        # E-step: compute expected moments for latent variable z.
        # -------------------------------------------------------
        Ez = self.E_z_given_y(Lambda, PLL_inv, y)
        Ezz = self.E_zzT_given_y(Lambda, PLL_inv, y, k)

        # M-step: compute optimal Lambda and Psi.
        # ---------------------------------------

        # Compute Lambda_new (Equation 5, G&H 1996).
        Lambda_lterm = LA.sum_outers(y, Ez)
        # Lambda_rterm = Ezz
        Lambda_new = Lambda_lterm @ inv(Ezz)

        # Compute Psi_diag_new (Equation 6, G&H 1996). Must use Lambda_new!
        Psi_rterm = LA.sum_outers(y, y) - LA.sum_outers(Lambda_new @ Ez, y)
        Psi_diag_new = 1. / n * diag(Psi_rterm)

        return Lambda_new, Psi_diag_new

# ------------------------------------------------------------------------------

    def E_z_given_y(self, L, PLL_inv, y):
        """Computes the first moment (mean) of the latent variable z.

        See Equation 2 in Ghahramani and Hinton (1996).

        :param L:       Current Lambda.
        :param PLL_inv: The inverse of (Psi + Lambda Lambda^T).
        :param y:       Observations of shape (n_dimensions, n_features).
        :return:        Expectation of z given y.
        """
        assert len(PLL_inv.shape) == 2
        beta = L.t() @ PLL_inv
        return beta @ y

# ------------------------------------------------------------------------------

    def E_zzT_given_y(self, L, PLL_inv, y, k):
        """Computes the second moment (variance) of the latent variable z.

        See Equation 4 in Ghahramani and Hinton (1996).

        :param L:       Current Lambda.
        :param PLL_inv: The inverse of (Psi + Lambda Lambda^T).
        :param y:       Observations of shape (n_dimensions, n_features).
        :param k:       Dimension of latent variable z.
        :return:        Expectation of the variance of z given y.
        """
        _, n = y.shape
        assert len(PLL_inv.shape) == 2
        beta = L.t() @ PLL_inv
        I    = torch.eye(k, device=device)
        bL   = beta @ L
        by   = beta @ y
        byyb = torch.einsum('ib,ob->io', [by, by])
        return n * (I - bL) + byyb

# ------------------------------------------------------------------------------

    def estimate_z_given_y(self, y, threshold):
        """Computes the first moment (mean) of the latent variable z using
        fitted parameters.

        :param y: Observations of shape (n_features, n_samples).
        :return:  Estimated expectation of z (n_features, n_samples).
        """
        k = 3 * self.latent_dim if self.private_z else self.latent_dim
        if threshold:
            inds = torch.abs(self.Lambda2) < threshold
            self.Lambda2.data[inds] = 0
            inds = torch.abs(self.B2) < threshold
            self.B2.data.data[inds] = 0
        Lambda, Psi_diag = self.tile_params()

        # if self.use_svd:
        # z = self.transform(y)
        # else:
        PLL_inv = LA.woodbury_inv(Psi_diag, Lambda, Lambda.t(), k)
        z = self.E_z_given_y(Lambda, PLL_inv, y)
        return z

# ------------------------------------------------------------------------------

    def sample(self, y, n_samples=None, one_sample_per_y=False):
        """Sample from the fitted probabilistic CCA model.

        :param y:         Observations of shape (n_features, n_samples).
        :param n_samples: The number of samples.
        :return:          Two views of n samples each.
        """
        k = 3 * self.latent_dim if self.private_z else self.latent_dim
        if one_sample_per_y:
            if n_samples and n_samples != y.shape[1]:
                msg = 'When sampling once per `y`, `n_samples` must be the' \
                      'number of samples of `y`.'
                raise AttributeError(msg)
            n_samples = y.shape[1]

        Lambda, Psi_diag = self.tile_params()

        # if self.use_svd:
        # z = self.transform(y)
        # else:
        PLL_inv = LA.woodbury_inv(Psi_diag, Lambda, Lambda.t(), k)
        z = self.E_z_given_y(Lambda, PLL_inv, y)

        # Note that given the structure of Lambda, this accounts for view-
        # specific information if `private_z` is `True`.
        m = Lambda @ z
        m1 = m[:self.p1]
        m2 = m[self.p1:]

        y1r = torch.empty(self.p1, n_samples, device=device)
        y2r = torch.empty(self.p2, n_samples, device=device)

        for i in range(n_samples):
            if one_sample_per_y:
                # Sample based on the estimated mean for the current `y`.
                j = i
            else:
                # Sample based on a randomly chosen latent variable.
                j = random.randint(0, z.shape[1]-1)
            y1r[:, i] = MVN(m1[:, j], diag(exp(self.log_Psi1_diag))).sample()
            y2r[:, i] = MVN(m2[:, j], diag(exp(self.log_Psi2_diag))).sample()

        return y1r.t(), y2r.t()

# ------------------------------------------------------------------------------

    def neg_log_likelihood(self, y, Lambda=None, Psi_diag=None):
        """Appendix A (p. 5) in Ghahramani and Hinton (1996).

        :param y:        (n x p)-dimensional observations.
        :param Lambda:   Current value for Lambda parameter.
        :param Psi_diag: Current value for Psi parameter.
        :return:         The negative log likelihood of the parameters given y.
        """
        p, n = y.shape
        k = 3 * self.latent_dim if self.private_z else self.latent_dim

        if Lambda is None and Psi_diag is None:
            Lambda, Psi_diag = self.tile_params()

        PLL_inv = LA.woodbury_inv(Psi_diag, Lambda, Lambda.t(), k)
        Ez  = self.E_z_given_y(Lambda, PLL_inv, y).t()
        Ezz = self.E_zzT_given_y(Lambda, PLL_inv, y, k).t()

        inv_Psi = diag(LA.diag_inv(Psi_diag))

        A = 1/2. * diag(y.t() @ inv_Psi @ y)
        B = diag(y.t() @ inv_Psi @ Lambda @ Ez.t())
        C = 1/2. * tr(Lambda.t() @ inv_Psi @ Lambda @ Ezz)
        rterm_sum = (A - B).sum() + C

        # This computes: `logdet = -n/2. * log(det(diag(Psi_diag)))`. For why,
        # see: http://gregorygundersen.com/blog/2019/01/18/log-likelihood/.
        logdet = -n/2. * log(Psi_diag).sum()

        ll = (logdet - rterm_sum).item()
        nll = -ll
        return nll

# ------------------------------------------------------------------------------

    def reparameterize(self, Lambda, Psi_diag, z):
        """Performs the reparameterization trick for a Gaussian random variable.
        For details, see:

            http://blog.shakirm.com/2015/10/
            machine-learning-trick-of-the-day-4-reparameterisation-tricks/

        :param Lambda:   Current value for Lambda parameter.
        :param Psi_diag: Current value for Psi parameter.
        :param z:        Latent variable.
        :return:         Samples of y from estimated parameters Lambda and Psi.
        """
        n = z.shape[1]
        p = Psi_diag.shape[0]
        eps = torch.randn(p, n, device=cuda.device())
        # For numerical stability. For Psi to be PD, all elements must be
        # positive: https://math.stackexchange.com/a/927916/159872.
        Psi_diag = LA.to_positive(Psi_diag)
        R = torch.potrf(diag(Psi_diag), upper=False)
        return Lambda @ z + R @ eps

# ------------------------------------------------------------------------------

    def init_params(self):
        """Create model parameters and move them to appropriate device.

        :return: Model parameters.
        """
        p1 = self.p1
        p2 = self.p2
        k  = self.latent_dim

        Lambda1 = torch.randn(p1, k).to(device)
        Lambda2 = torch.randn(p2, k).to(device)

        # Create these modality-specific parameters regardless. We will discard
        # them if the user did not set `private_z=True`.
        B1 = torch.randn(p1, k).to(device)
        B2 = torch.randn(p2, k).to(device)

        log_Psi1_diag = torch.ones(p1).to(device)
        log_Psi2_diag = torch.ones(p2).to(device)

        return Lambda1, Lambda2, B1, B2, log_Psi1_diag, log_Psi2_diag

# ------------------------------------------------------------------------------

    def tile_params(self):
        """Tile parameters so we can use factor analysis updates for PCCA.

        :return: Model parameters concatenated appropriately.
        """
        p1 = self.p1
        p2 = self.p2
        k  = self.latent_dim

        B12 = torch.zeros(p1, k).to(device)
        B21 = torch.zeros(p2, k).to(device)

        if self.private_z:
            Lambda = torch.cat([
                torch.cat([self.Lambda1, self.B1, B12], dim=1),
                torch.cat([self.Lambda2, B21, self.B2], dim=1)
            ], dim=0)
        else:
            Lambda = torch.cat([self.Lambda1, self.Lambda2], dim=0)

        log_Psi_diag = torch.cat([self.log_Psi1_diag, self.log_Psi2_diag])
        Psi_diag     = exp(log_Psi_diag)

        return Lambda, Psi_diag

# ------------------------------------------------------------------------------

    def untile_params(self, Lambda, log_Psi_diag):
        """Takes tiled parameters and untiles them; reverse of `tile_params()`.

        :return: Model parameters unrolled or un-concatenated appropriately.
        """
        k = self.latent_dim
        Lambda1 = Lambda[:self.p1, :k]
        Lambda2 = Lambda[self.p1:, :k]
        log_Psi1_diag = log_Psi_diag[:self.p1]
        log_Psi2_diag = log_Psi_diag[self.p1:]

        if self.private_z:
            B1 = Lambda[:self.p1, k:2 * k]
            B2 = Lambda[self.p1:, 2 * k:]
            return Lambda1, Lambda2, B1, B2, log_Psi1_diag, log_Psi2_diag
        else:
            return Lambda1, Lambda2, log_Psi1_diag, log_Psi2_diag

# ------------------------------------------------------------------------------

    def save_new_params(self, Lambda_new, log_Psi_diag_new):
        """Save new parameters after EM.

        :param Lambda_new:   Lambda after EM.
        :param log_Psi_diag_new: Psi_diag after EM.
        """
        assert not self.differentiable

        params = self.untile_params(Lambda_new, log_Psi_diag_new)

        if self.private_z:
            Lambda1, Lambda2, B1, B2, log_Psi1_diag, log_Psi2_diag = params
            self.Lambda1 = Lambda1.detach()
            self.Lambda2 = Lambda2.detach()
            self.B1 = B1.detach()
            self.B2 = B2.detach()
            self.log_Psi1_diag = log_Psi1_diag.detach()
            self.log_Psi2_diag = log_Psi2_diag.detach()
        else:
            Lambda1, Lambda2, log_Psi1_diag, log_Psi2_diag = params
            self.Lambda1 = Lambda1.detach()
            self.Lambda2 = Lambda2.detach()
            self.log_Psi1_diag = log_Psi1_diag.detach()
            self.log_Psi2_diag = log_Psi2_diag.detach()

# ------------------------------------------------------------------------------

    # PyTorch will place a method `parameters` on our class. Don't step on it.
    def parameters_(self, view=None):
        """Generates an iterator of parameters for the PCCA instance.

        :param view: An optional string specifying which data type's
                     parameters to return.
        :return:     The aforementioned iterator.
        """
        if not view:
            params = [self.Lambda1, self.Lambda2, self.log_Psi1_diag,
                      self.log_Psi2_diag]
            if self.private_z:
                params[2:2] = [self.B1, self.B2]
        elif view == 'y1':
            params = [self.Lambda1]
            if self.private_z:
                params.append(self.B1)
        elif view == 'y2':
            params = [self.Lambda2]
            if self.private_z:
                params.append(self.B2)
        else:
            raise AttributeError('Supported modalities: "y1", "y2".')

        return iter(params)
