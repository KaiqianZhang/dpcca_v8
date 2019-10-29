"""=============================================================================
Sanity check that images look normal:

- Before normalization.
- After normalization.
- After subsampling.
============================================================================="""

import csv
from data.gtex import GTExDataset, config
import glob
from PIL import Image
import numpy as np
import random
import torch
from torchvision import utils

# ------------------------------------------------------------------------------

OUTPUT_DIRECTORY = 'scratch'
N_IMAGES = 100


# ------------------------------------------------------------------------------

def visualize_images(data, subdir):
    for i in range(N_IMAGES):
        r = random.randint(0, len(data['images']) - 1)
        image = data['images'][r]
        tissue = data['tissues'][r]
        fname = '%s/%s/%s_%s.png' % (OUTPUT_DIRECTORY, subdir, tissue, i)
        utils.save_image(image, fname)


# ------------------------------------------------------------------------------

def visualize_original_images():
    # First, build an associative array of classes.
    name_to_tissue = {}
    with open('%s/GTEx_classes.txt' % config.ROOT_DIR) as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            image_name = row[0].replace('.jpg', '')
            class_ = row[1]
            name_to_tissue[image_name] = class_

    # Next, grab some images and find their classes.
    i = 0
    for fname in glob.glob('%s/*.jpg' % '%s/images' % config.ROOT_DIR):
        img = Image.open(fname).convert('RGB')
        img = np.array(img).T
        img_data = torch.Tensor(img.tolist())
        key = fname.replace('.jpg', '').replace('data/gtex/images/', '')
        tissue_type = name_to_tissue[key]
        # See comment above PyTorch's `transforms.Normalize` for why we do this.
        # The upshot: need to set image to between 0 and 1.
        img_data /= 255.0
        fname = '%s/%s/%s_%s.png' % (OUTPUT_DIRECTORY, 'original', i,
                                     tissue_type)
        utils.save_image(img_data, fname)
        i += 1
        if i > N_IMAGES:
            return


# ------------------------------------------------------------------------------

def visualize_subsampled_images():
    dataset = GTExDataset(subsample=False)
    indices = np.random.permutation(len(dataset))[:N_IMAGES]
    data = [dataset[i] for i in indices]
    for i in range(len(indices)):
        fname = '%s/%s/%s_%s.png' % (OUTPUT_DIRECTORY, 'subsampled',
                                     data[i]['tissue'], i)
        # image = transform.
        utils.save_image(data[i]['image'], fname)


# ------------------------------------------------------------------------------

def visualize_normalized_images():
    data = torch.load('data/gtex/train.pth')
    subdir = 'normalized'
    visualize_images(data, subdir)


# ------------------------------------------------------------------------------

if __name__ == '__main__':
    visualize_subsampled_images()
