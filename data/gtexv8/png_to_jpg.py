import os
from   PIL import Image
import torch

for i, fname in enumerate(next(os.walk('images'))[2]):
    im     = Image.open('images/%s' % fname)
    rgb_im = im.convert('RGB')
    base   = fname.split('.')[0]
    rgb_im.save('images/%s.jpg' % base)
    if i % 100 == 0:
        print(i)

data = torch.load('train.pth')
data['images'] = data['images']['filename'].str.replace('.png', '.jpg')
torch.save(data, 'train.pth')
