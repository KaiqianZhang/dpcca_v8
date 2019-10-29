"""=============================================================================
Create sub-dataset of a specific tissue
============================================================================="""

import sys

import torch

from data.gtex.config import GTExDataset, GTExConfig
cfg = GTExConfig()

# ------------------------------------------------------------------------------

def select_samples(tissue):
    dataset = GTExDataset(cfg)

    count = 0
    for tiss in dataset.tissues:
        if tiss == tissue:
            count += 1

    images  = torch.Tensor(count, 3, 512, 512)
    genes   = torch.Tensor(count, 18659)
    names   = []
    tissues = []

    i = 0
    for img, gene, name, tiss in zip(dataset.images, dataset.genes,
                                     dataset.names, dataset.tissues):
        if tiss == tissue:
            images[i] = img
            genes[i] = gene
            names.append(name)
            tissues.append(tiss)
            i += 1

    return images, genes, names, tissues

# ------------------------------------------------------------------------------

if __name__ == '__main__':
    tissue = sys.argv[1].capitalize()
    images, genes, names, tissues = select_samples(tissue)
    torch.save({
        'names': names,
        'tissues': tissues,
        'images': images,
        'genes': genes
    }, '%s/%s.pth' % (cfg.ROOT_DIR, tissue.lower()))
