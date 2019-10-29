"""=============================================================================
GTEx data preprocessing script.
============================================================================="""

import gc
import glob
import pandas as pd
from   PIL import Image
import numpy as np
import torch
from   torchvision import transforms

from data.gtex.config import GTExConfig
config = GTExConfig()


# ------------------------------------------------------------------------------

def preprocess():
    """Return Torch tensors of data after processing JPEG files and parsing
    gene expression data.
    """
    print('Loading and quantile normalizing genes.')
    g_fnames, g_names, values = _preprocess_genes()

    print('Loading and preprocessing images.')
    i_fnames, images = _preprocess_images()

    # Resize images before sampling. Based on Jordan et al's work.
    # NEW_SIZE = 512
    # print('Compressing images to size: %s x %s.' % (NEW_SIZE, NEW_SIZE))
    # images = _compress_images(images, NEW_SIZE)

    print('Loading and processing labels.')
    l_names, tissues = _get_ordered_tissues()

    assert (g_fnames == i_fnames).all()
    assert (i_fnames == l_names).all()
    print('Data sorted correctly.')

    print('Saving images.')
    genes  = torch.Tensor(values)
    images = torch.Tensor(images)
    _save(g_fnames, tissues, images, genes, g_names)


# ------------------------------------------------------------------------------

def _preprocess_genes():
    """Return Torch tensor of GTEx gene expression data.
    """
    GENES_FPATH = '%s/RNA-Seq_GTEx.txt' % config.ROOT_DIR
    df = pd.read_csv(GENES_FPATH, delimiter='\t')

    df = _quant_normalize(df)

    df.sort_index(inplace=True)

    fnames = np.array(df.index.tolist())
    gnames = df.columns
    values = df.values

    values = _standardize_range_0_to_1(values)
    assert values.min() == 0
    assert values.max() == 1
    # A brute force sanity check that the numbers aren't _all_ 1s or 0s.
    assert ((values != 0) == (values != 1)).sum() > 0

    return fnames, gnames, values


# ------------------------------------------------------------------------------

def _preprocess_images():
    """Return Torch tensor of imaging data.
    """
    IMG_PATH = '%s/images' % config.ROOT_DIR
    images = np.zeros((config.N_SAMPLES, config.N_CHANNELS, 1000, 1000))
    fnames = []

    for i, fname in enumerate(glob.glob('%s/*.jpg' % IMG_PATH)):
        print(fname)
        if i % 100 == 0:
            n = gc.collect()
            print(i, n)

        img = Image.open(fname).convert('RGB')
        img = np.array(img).T
        img = torch.Tensor(img.tolist())
        # See comment above PyTorch's `transforms.Normalize` for why we do this.
        # The upshot: need to set image to between 0 and 1.
        img /= 255.0
        images[i] = img.numpy()

        label = fname.split('/')[-1].replace('.jpg', '')
        fnames.append(label)

    fnames  = np.array(fnames)
    indices = np.argsort(fnames)
    fnames  = fnames[indices]
    images  = images[indices]

    return fnames, images


# ------------------------------------------------------------------------------

def _compress_images(images, new_size):
    """Compress images now rather than for every single image for every single
    batch.
    """
    n_samples, n_channels, _, _ = images.shape
    new_images = torch.Tensor(n_samples, n_channels, new_size, new_size)
    subsample = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(new_size),
        transforms.ToTensor()
    ])
    for i, img in enumerate(torch.Tensor(images)):
        if i % 10 == 0:
            print(i)
        new_images[i] = subsample(img)
    return new_images


# ------------------------------------------------------------------------------

def _get_ordered_tissues():
    """Order labels by file name a la the other functions.
    """
    fname = '%s/GTEx_classes.txt' % config.ROOT_DIR
    df = pd.read_csv(fname, delimiter='\t', header=None, index_col=0)
    df.index = df.index.str.replace('.jpg', '')
    # Sort by tissue.
    df.sort_index(inplace=True)
    names = np.array(df.index.tolist())
    tissues = np.array(df[1].tolist())
    return names, tissues


# ------------------------------------------------------------------------------

def _save(fnames, tissues, images, genes, gnames):
    """Save images and genes to Torch tensor.
    """
    out_dir = '%s/train_1000x1000.pth' % config.ROOT_DIR
    torch.save({
        'fnames':  fnames,  # File names, e.g. GTEX-XMK1-1026.jpg.
        'tissues': tissues, # Tissue names.
        'images':  images,  # Image pixel values.
        'genes':   genes,   # Gene expression values.
        'gnames':  gnames   # Gene names, i.e. FOXE1
    }, out_dir)


# ------------------------------------------------------------------------------

def _standardize_range_0_to_1(genes):
    """Return genes to be standardized between 0 and 1.
    """
    max_ = genes.max()
    min_ = genes.min()
    return (genes - min_) / (max_ - min_)


# ------------------------------------------------------------------------------

def _quant_normalize(df_input):
    """Quantile normalization is a 4 step algorithm to make two or more
    distributions identical in statistical properties. Below is a visualization:

             Original    Ranked    Averaged    Re-ordered
             A   B       A   B     A   B       A   B
    gene1    2 4 8 6     2 4 3 3   3 3 3 3     3 3 6 6
    gene2    6 4 3 3     6 4 8 6   6 6 6 6     6 6 3 3

    Read more here: http://en.wikipedia.org/wiki/Quantile_normalization
    """
    # Generate a new DF with each column sorted from least to greatest.
    df = df_input.copy()
    dic = {}
    for gene in df:
        dic.update({gene: sorted(df[gene])})
    sorted_df = pd.DataFrame(dic)

    # Get the rank of each row, which is just the average.
    rank = sorted_df.mean(axis=1).tolist()

    # Re-order the data.
    for gene in df:
        # This just gives us the indices by which to re-order the data:
        #
        # >>> np.searchsorted([1,2,3,4], [4,1,3,2])
        # array([3, 0, 2, 1])
        #
        t = np.searchsorted(np.sort(df[gene]), df[gene])
        df[gene] = [rank[i] for i in t]

    max_ = df.max().max()
    min_ = df.min().min()
    for gene in df:
        try:
            assert df[gene].max() == max_
            assert df[gene].min() == min_
        except AssertionError:
            # If the above assertion fails, it is because every value of the
            # gene is 0.
            assert (df_input[gene] == 0).all()

    return df


# ------------------------------------------------------------------------------

if __name__ == '__main__':
    preprocess()
