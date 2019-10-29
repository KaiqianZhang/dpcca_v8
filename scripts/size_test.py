from   models import DCGANVAE128
from   data import GTExV8Config
import torch

cfg = GTExV8Config()
model = DCGANVAE128(cfg)

x1 = torch.ones(100, 3, 128, 128)
x1r = model.forward(x1)

assert x1.shape == x1r.shape