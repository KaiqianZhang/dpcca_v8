"""=============================================================================
Generate a heatmap of gene expression levels in an image with pixel-level
resolution.
============================================================================="""

from   PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random
import seaborn as sns
import torch
from   torchvision import transforms as T

from   data import GTExConfig, GTExDataset
from   models import DPCCA

# ------------------------------------------------------------------------------

def main(tissue, gene_name):
    cfg = GTExConfig()
    dataset = GTExDataset(cfg)

    path = 'experiments/20181014_celebaae28_1000x1000/'\
           'gtex_mode-ccaae_latent_dim-100_seed-0_lr-0.0001/model.pt'

    state = torch.load(path)
    model = DPCCA(cfg, latent_dim=100, private_z=True)
    model.load_state_dict(state)

    image, r_idx = get_image(dataset, tissue)
    gene_idx = list(dataset.gene_names).index(gene_name)

    heatmap = gen_heatmap(model, gene_idx, image, n_for_avg=1)

    fig, ax = plt.subplots()
    fig.set_size_inches(15, 15)
    heatmap = np.array(heatmap)
    ax = sns.heatmap(heatmap, vmin=0, vmax=1)
    plt.savefig('%s_%s_idx-%s.png' % (tissue, gene_name, r_idx))

# ------------------------------------------------------------------------------

def gen_heatmap(model, gene_idx, image, n_for_avg=1):
    heatmap = []

    ii = 0
    for i in range(0, 980, 14):
        heatmap.append([])
        for j in range(0, 980, 14):
            x1 = image[:, i:i + 28, j:j + 28]
            val_avg = []
            for _ in range(n_for_avg):
                x2r = model.sample_x2_from_x1(torch.Tensor(x1).unsqueeze(0))
                gene_val = x2r.squeeze()[gene_idx]
                val_avg.append(gene_val.item())
            heatmap[ii].append(np.array(val_avg).mean())
        ii += 1
        print('Done %s' % i)
    print('Heatmap completed.')
    return heatmap

# ------------------------------------------------------------------------------

def get_image(dataset, tissue):

    inds   = (dataset.tissues == tissue).astype(int)
    inds   = torch.Tensor(inds).byte()
    images = dataset.images[inds]
    names  = dataset.names[inds]

    r = random.randint(0, len(images))
    print('Sample index: %s' % r)

    resize_for_viz = T.Compose([
        T.ToPILImage(),
        T.Resize(300),
        T.ToTensor()
    ])

    img_4_viz = resize_for_viz(images[r]).numpy() * 255
    img_4_viz = img_4_viz.astype('uint8')
    img_4_viz = Image.fromarray(img_4_viz.T)
    img_4_viz.save('%s.png' % names[r])

    return images[r], r

# ------------------------------------------------------------------------------

if __name__ == '__main__':
    main('Thyroid', 'FOXE1')
