"""============================================================================
Script to explore PyTorch's upsampling options.
============================================================================"""

import numpy as np
from   PIL import Image
import torch
from   torch import nn
from   torchvision.utils import save_image
from   torchvision import transforms

# ------------------------------------------------------------------------------

resize = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(50),
    transforms.ToTensor()
])

SCALE = 20

# ------------------------------------------------------------------------------

fname = 'data/gtex/images/GTEX-13PL7-2326.jpg'
img   = Image.open(fname).convert('RGB')
img   = np.array(img).T
img   = torch.Tensor(img.tolist())

img /= img.max()
img = img.unsqueeze(0)

save_image(img, 'scripts/upsampling/orig.png')

# Start with a smaller image and then upsample.
img = resize(img.squeeze(0)).unsqueeze(0)
save_image(img, 'scripts/upsampling/small.png')

m = nn.Upsample(scale_factor=SCALE, mode='nearest')
save_image(m(img), 'scripts/upsampling/nearest.png')

m = nn.Upsample(scale_factor=SCALE, mode='bilinear', align_corners=False)
save_image(m(img), 'scripts/upsampling/bilinear.png')

m = nn.Upsample(scale_factor=SCALE, mode='bilinear', align_corners=True)
save_image(m(img), 'scripts/upsampling/bilinear_align_corners.png')

# m = nn.Upsample(scale_factor=2, mode='linear')
# save_image(m(img), 'scripts/upsampling/linear.png', align_corners=False)

