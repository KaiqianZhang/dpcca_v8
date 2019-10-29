"""=============================================================================
Perform some standardized analysis on a trained model.
============================================================================="""

from   torch.nn import functional as F
from   sklearn.cluster import KMeans
from   sklearn.metrics import silhouette_score

import cuda
import pprint

# ------------------------------------------------------------------------------

device = cuda.device()

# ------------------------------------------------------------------------------

def log_metrics(model, dataset):
    """Log all model metrics.
    """
    x1 = dataset.images.to(device)
    x2 = dataset.genes.to(device)
    x1r, x2r = model.forward([x1, x2])

    pprint.log('Reconstruction loss    : %s' % reconstruction(x1, x2, x1r, x2r))
    pprint.log('Negative log likelihood: %s' % neg_log_likelihood(model, [x1, x2]))
    pprint.log('Silhouette score       : %s' % clustering(model, dataset,
                                                     dataset.n_classes))

# ------------------------------------------------------------------------------

def reconstruction(x1, x2, x1r, x2r):
    """Given a model and some data, forward the model
    """
    return (F.mse_loss(x1r, x1) + F.mse_loss(x2r, x2)).item()

# ------------------------------------------------------------------------------

def neg_log_likelihood(model, x):
    """Compute the negative log likelihood of the embeddings given data.
    """
    return model.neg_log_likelihood(x).item()

# ------------------------------------------------------------------------------

def clustering(model, dataset, n_clusters):
    """Computes the Silhouette score for the estimated latent variables after
    labeling the cluters with K-means.

    See: https://en.wikipedia.org/wiki/Silhouette_(clustering).
    """
    x1 = dataset.images
    x2 = dataset.genes
    z = model.estimate_z_given_x([x1, x2]).detach()
    fuzzy_label = KMeans(n_clusters=n_clusters).fit_predict(z)
    return silhouette_score(z.detach(), fuzzy_label)
