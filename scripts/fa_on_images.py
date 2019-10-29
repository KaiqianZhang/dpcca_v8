"""=============================================================================
Script to verify that factor analysis works on imaging data.
============================================================================="""

import numpy as np
from   models.fa import FA
import random
from   sklearn.decomposition import FactorAnalysis
import torch
import torchvision.datasets as datasets
from   torchvision import transforms
from   torchvision.utils import save_image
from   torch.distributions.multivariate_normal import MultivariateNormal as MVN
from   data import GTExV8Config, GTExV8Dataset

# ------------------------------------------------------------------------------

# data = datasets.MNIST(root='mnist',
#                       train=True,
#                       transform=transforms.ToTensor(),
#                       download=True)
# data_dim = np.prod(data.train_data[0].shape)
# data = data.train_data.view(-1, 28*28)[:10000].float()
# data = data / 255.
# data = data - data.mean(dim=0)

cfg = GTExV8Config()
dataset = GTExV8Dataset(cfg)

n_samples = 1000
y = torch.Tensor(n_samples, 3, 128, 128)
for i in range(n_samples):
    y1, _ = dataset[i]
    y[i, :] = y1

# model = FA(latent_dim=2, data_dim=y.shape[1], n_iters=100)
# model.fit(y.t())
# images = model.sample(y.t(), 200)
# for i, img in enumerate(images.t()):
#     img = img.view(1, 28, 28)
#     save_image(img, '%s.png' % i)

y = y.view(-1, 3*128*128)
model = FactorAnalysis(n_components=100, max_iter=1000)
z = model.fit_transform(y)

Lambda = model.components_.T
means = Lambda @ z.T
Psi = model.noise_variance_

y = torch.empty(y.shape)

for i in range(30):
    r = random.randint(0, len(means))
    img = MVN(torch.Tensor(means[:, i]), torch.diag(torch.Tensor(Psi))).sample()
    img = img.view(3, 128, 128)
    save_image(img, '%s.png' % i)

