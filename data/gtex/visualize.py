"""=============================================================================
Create small dataset with just muscle, heart, brain, and pancreas samples.
============================================================================="""

import torch
from   torchvision.utils import save_image

# ------------------------------------------------------------------------------'

data = torch.load('data/gtex/train_four_tissues.pth')

images  = data['images']
tissues = data['tissues']

for i, (img, tis) in enumerate(zip(images, tissues)):
    fname = 'scratch/four_tissues/%s-%s.png' % (i, tis)
    save_image(img, fname)
