"""=============================================================================
Cluster Z based on experiment directory
============================================================================="""

import os
import re
import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn
from   sklearn.decomposition import PCA
from   sklearn.manifold import TSNE
import torch

from   data import GTExConfig, GTExDataset
from   models import DPCCA

# ------------------------------------------------------------------------------

cfg     = GTExConfig()
dataset = GTExDataset(cfg)

# ------------------------------------------------------------------------------

def main(directory):
    root = '/Users/gwg/local/dmcm/experiments/%s' % directory
    for subdir, dirs, files in os.walk(root):
        print(subdir)
        for f in files:
            if f != 'model.pt':
                continue

            fname = subdir + '/' + f

            search = re.search('latent_dim-[0-9]*', fname)
            latent_dim = int(search.group().split('-')[1])

            model = load_model(fname, latent_dim)
            z, zs, z1, z2, df, labels = estimate_z(model)

            df.to_csv(subdir + '/estimated_z.csv')
            print('Z estimated and saved.')
            plot(z,  labels, 'all')
            plot(zs, labels, 'shared')
            plot(z1, labels, 'images')
            plot(z2, labels, 'genes')
            plot(np.hstack([zs, z1]), labels, 'shared-and-images')
            plot(np.hstack([zs, z2]), labels, 'shared-and-genes')

            # For now, only cluster using the first model.
            return

# ------------------------------------------------------------------------------

def load_model(fname, latent_dim):
    state_dict = torch.load(fname, map_location={'cuda:0': 'cpu'})
    cfg.PCCA_Z_DIM = latent_dim
    model = DPCCA(cfg)
    model.load_state_dict(state_dict)
    return model

# ------------------------------------------------------------------------------

def estimate_z(model):
    n = 500
    x1_batch = torch.Tensor(n, cfg.N_CHANNELS, cfg.IMG_SIZE, cfg.IMG_SIZE)
    x2_batch = torch.Tensor(n, cfg.N_GENES)

    labels = []
    names = []
    for i in range(n):
        x1, x2 = dataset[i]
        x1_batch[i, :] = x1
        x2_batch[i, :] = x2
        labels.append(int(dataset.labels[i].item()))
        names.append(dataset.names[i])

    z = model.estimate_z([x1_batch, x2_batch])
    z = z.detach().numpy()

    k  = 100
    zs = z[:, :k]
    z1 = z[:, k:2*k]
    z2 = z[:, 2*k:]

    tissues = dataset.labelEncoder.inverse_transform(labels)
    df = pd.DataFrame(data=z, index=[names, tissues])

    labels = np.array(labels)
    return z, zs, z1, z2, df, labels

# ------------------------------------------------------------------------------

def plot(z, labels, desc):
    pca  = PCA(n_components=50)
    z = pca.fit_transform(z)
    print('PCA fit.')
    tsne = TSNE(n_components=2)
    z = tsne.fit_transform(z)
    print('t-SNE fit.')

    fig, ax = plt.subplots(1, 1, dpi=72)
    fig.set_size_inches(20, 10)  # Width, height

    clrs = seaborn.color_palette('hls', n_colors=len(dataset.classes))
    LINE_STYLES = ['o', 'v', 's', '*']
    NUM_STYLES = len(LINE_STYLES)

    Xp = z[:, 0]
    Yp = z[:, 1]

    for i in range(len(dataset.classes)):
        indices = labels == i
        x = Xp[indices]
        y = Yp[indices]
        label = dataset.labelEncoder.inverse_transform([i])[0]
        marker = LINE_STYLES[i % NUM_STYLES]
        ax.scatter(x, y, c=clrs[i], label=label, marker=marker, zorder=10)

    plt.legend()
    plt.savefig('/Users/gwg/Desktop/%s.png' % desc)
    # plt.show()

# ------------------------------------------------------------------------------

if __name__ == '__main__':
    directory = sys.argv[1]
    main(directory)