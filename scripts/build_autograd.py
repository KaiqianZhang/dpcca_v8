"""=============================================================================
Script for building the autograd graph given our model on MNIST data.
============================================================================="""

import os
import torch
from   torchviz import make_dot
from   data import loader
from   models import DPCCA

# ------------------------------------------------------------------------------

cfg = loader.get_config('mnist')

model = DPCCA(cfg, latent_dim=2, em_iters=1)

x1 = torch.randn(128, 1, 28, 28)
x2 = torch.randn(128, 100)

params = dict(model.named_parameters())
dot = make_dot(model([x1, x2]), params=params)
dot.render('autograd')

os.remove('/Users/gwg/dmcm/autograd')
