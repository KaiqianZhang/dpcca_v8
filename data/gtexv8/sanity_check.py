"""=============================================================================
Verify that images by tissue label appear sensible.
============================================================================="""

from   PIL import Image
import random
import torch

# ------------------------------------------------------------------------------

data = torch.load('train.pth')
n    = len(data['samples'])

# ------------------------------------------------------------------------------

for i in range(100):
    r      = random.randint(0, n - 1)
    jpg    = data['images'].iloc[r]
    tiss   = data['tissues'].iloc[r][0]
    impath = 'images/%s' % jpg
    img    = Image.open(impath)
    img.save('check/%s_%s.jpg' % (i, tiss))
    print(i)
