"""=============================================================================
Sanity check that synthetic images can be printed reasonably.
============================================================================="""

from torchvision import utils
from data import ToyDataset

# ------------------------------------------------------------------------------

def visualize_images():
    dataset = ToyDataset()
    images = dataset.images
    for i in range(100):
        image = images[i]
        fname = 'scratch/synth/%s.png' % i
        utils.save_image(image, fname)

# ------------------------------------------------------------------------------

if __name__ == '__main__':
    visualize_images()
