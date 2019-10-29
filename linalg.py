"""=============================================================================
Functions for linear algebra operations.
============================================================================="""

import cuda
import torch

# ------------------------------------------------------------------------------

diag = torch.diag
inv  = torch.inverse

# ------------------------------------------------------------------------------

def to_positive(A_diag, eps=0.00001):
    """Convert n-vector into an n-vector with nonnegative entries.
    """
    A_diag[A_diag < 0] = eps
    inds = torch.isclose(A_diag, torch.zeros(1, device=cuda.device()))
    A_diag[inds] = eps
    return A_diag

# ------------------------------------------------------------------------------

def woodbury_inv(A_diag, U, V, k):
    """This matrix inversion is O(k^3) rather than O(p^3) where p is the
    dimensionality of the data and k is the latent dimension. For details, see:

        http://gregorygundersen.com/blog/2018/11/30/woodbury/
    """
    # Helps with numerics. If A_diag[i, j] == 0, then 1 / 0 == inf.
    SMALL = 1e-12
    A_inv_diag = 1. / (A_diag + SMALL)

    I     = torch.eye(k, device=cuda.device())
    B_inv = inv(I + ((V * A_inv_diag) @ U))

    # We want to perform the operation `U @ B_inv @ V` but need to optimize it:
    # - Computing `tmp1` is fast because it is (p, k) * (k, k).
    # - Computing `tmp2` is slow because it is (p, k) * (k, p).
    tmp1  = U @ B_inv
    tmp2  = torch.einsum('ab,bc->ac', (tmp1, V))

    # Use `view` rather than `reshape`. The former guarantees that a new tensor
    # is returned.
    tmp3  = A_inv_diag.view(-1, 1) * tmp2
    right = tmp3 * A_inv_diag

    # This is a fast version of `diag(A_inv_diag) - right`.
    right = -1 * right
    idx   = torch.arange(0, A_diag.size(0), device=cuda.device())
    right[idx, idx] = A_inv_diag + right[idx, idx]

    return right

# ------------------------------------------------------------------------------

def diag_inv(A):
    """The inverse of a diagonal matrix is just the reciprocal of each of its
    diagonal elements
    """
    return diag(1. / diag(A))

# ------------------------------------------------------------------------------

def sum_outers(x, y):
    """Return sum of outer products of paired columns of x and y.
    """
    # In PyTorch 4.0, `einsum` modifies variables inplace. This will not work
    # unless you have PyTorch 4.1:
    #
    #     https://github.com/pytorch/pytorch/issues/7763
    #
    return torch.einsum('ab,cb->ac', [x, y])

# ------------------------------------------------------------------------------

def rand_svd(A, rank, n_oversamples=None, n_subspace_iters=None):
    """Randomized SVD (p. 227 of Halko et al).

    :param A:                (m x n) matrix.
    :param rank:             Desired rank approximation.
    :param n_oversamples:    Oversampling parameter for Gaussian random samples.
    :param n_subspace_iters: Number of power iterations.
    :return:                 U, S, and Vt as in truncated SVD.
    """
    if n_oversamples is None:
        # This is the default used in the paper.
        n_samples = 2 * rank
    else:
        n_samples = rank + n_oversamples

    # Stage A.
    Q = find_range(A, n_samples, n_subspace_iters)

    # Stage B.
    B = Q.t() @ A
    U_tilde, S, Vt = torch.svd(B)
    U = Q @ U_tilde

    return U[:, :rank], S[:rank], Vt[:rank, :]

# ------------------------------------------------------------------------------

def find_range(A, n_samples, n_subspace_iters=None):
    """Algorithm 4.1: Randomized range finder (p. 240 of Halko et al).

    Given a matrix A and a number of samples, computes an orthonormal matrix
    that approximates the range of A.

    :param A:                (m x n) matrix.
    :param n_samples:        Number of Gaussian random samples.
    :param n_subspace_iters: Number of subspace iterations.
    :return:                 Orthonormal basis for approximate range of A.
    """
    m, n = A.shape
    O = torch.randn(n, n_samples)
    Y = A @ O

    if n_subspace_iters:
        return subspace_iter(A, Y, n_subspace_iters)
    else:
        return ortho_basis(Y)

# ------------------------------------------------------------------------------

def subspace_iter(A, Y0, n_iters):
    """Algorithm 4.4: Randomized subspace iteration (p. 244 of Halko et al).

    Uses a numerically stable subspace iteration algorithm to down-weight
    smaller singular values.

    :param A:       (m x n) matrix.
    :param Y0:      Initial approximate range of A.
    :param n_iters: Number of subspace iterations.
    :return:        Orthonormalized approximate range of A after power
                    iterations.
    """
    Q = ortho_basis(Y0)
    for _ in range(n_iters):
        Z = ortho_basis(A.t() @ Q)
        Q = ortho_basis(A @ Z)
    return Q

# ------------------------------------------------------------------------------

def ortho_basis(M):
    """Computes an orthonormal basis for a matrix.

    :param M: (m x n) matrix.
    :return:  An orthonormal basis for M.
    """
    Q, _ = torch.qr(M)
    return Q
