"""=============================================================================
Performance test between my implementation of factor analysis and sklearn's.
============================================================================="""

import time
from   models.fa import FA
from   sklearn.decomposition import FactorAnalysis
import torch

# ------------------------------------------------------------------------------

y = torch.randn(1000, 500)
k = 100

s = time.time()
FA(latent_dim=k, data_dim=y.shape[1], n_iters=100).fit(y.t())
print('My FA:                  %s' % str(time.time() - s))

s = time.time()
FactorAnalysis(n_components=k, max_iter=100).fit_transform(y.numpy())
print('sklearn\'s FA (rsvd):    %s' % str(time.time() - s))

s = time.time()
FactorAnalysis(n_components=k, max_iter=100, svd_method='lapack')\
    .fit_transform(y.numpy())
print('sklearn\'s FA (lapack):  %s' % str(time.time() - s))
