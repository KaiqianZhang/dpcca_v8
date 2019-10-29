"""=============================================================================
Script to embed data if it fails on cluster for any reason.
============================================================================="""

import torch
from   data import GTExV8Config, GTExV8Dataset
from   models import DPCCA
import cuda

# ------------------------------------------------------------------------------

device             = cuda.device()
cfg                = GTExV8Config()
cfg.IMG_EMBED_DIM  = 100
cfg.GENE_EMBED_DIM = 100
latent_dim         = 10

path = 'experiments/20190131_big_sweep/gtexv8_mode-dpcca_batch_size-128'\
       '_seed-0_latent_dim-10_lr-0.0001_l1_coef-0.5_em_iters-1_clip-1/model.pt'

# ------------------------------------------------------------------------------

state = torch.load(path, map_location={'cuda:0': 'cpu'})
model = DPCCA(cfg, latent_dim=10, use_gene_net=True)
model.load_state_dict(state)
model = model.to(device)

dataset = GTExV8Dataset(cfg)
print('Dataset loaded.')

n  = len(dataset)
X1 = torch.Tensor(n, cfg.N_CHANNELS, cfg.IMG_SIZE, cfg.IMG_SIZE)
X2 = torch.Tensor(n, cfg.N_GENES)
for i in range(n):
    x1, x2 = dataset[i]
    X1[i] = x1
    X2[i] = x2
print('Data created.')

X1 = X1.to(device)
X2 = X2.to(device)
Z  = model.estimate_z_given_x([X1, X2], threshold=None).detach()
print('Embeddings created.')

torch.save(Z, 'embeddings.pt')
