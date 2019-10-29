"""=============================================================================
resize_images.py
Gregory Gundersen
============================================================================="""

import torch
from torchvision import transforms as T
import PIL

# ------------------------------------------------------------------------------

# New image size.
NEW_SIZE = 100
resize = T.Compose([T.ToPILImage(),
                    T.Resize((NEW_SIZE, NEW_SIZE)),
                    T.ToTensor()])

print('Loading images.')
images = torch.load('data/gtex/images.pth')
print('Images loaded.')

small_images = torch.Tensor(images.size(0),
                            images.size(1),
                            NEW_SIZE,
                            NEW_SIZE)

print('Resizing images.')
for i, img in enumerate(images):
    small_images[i] = resize(img)
print('Images resized.')

xmin = small_images.min()
xmax = small_images.max()

small_images = (small_images - xmin) / (xmax - xmin)

print('Saving tensor')
torch.save(small_images, 'data/gtex/images_32x32.pth')
