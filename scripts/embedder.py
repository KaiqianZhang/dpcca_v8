"""=============================================================================
Embed data based on trained DMCM model.
============================================================================="""

import gc
import os

from   data.gtex.dataset import GTExImages

import torch
from   torch.autograd import Variable
from   torch.utils.data import DataLoader
from   torch.utils.data.sampler import SequentialSampler

import cuda

# ------------------------------------------------------------------------------

def embed(directory, config, model, mode):
    """Embed data by performing and saving single pass of dataset through
    trained model.
    """
    dataset = config.get_dataset()
    indices = list(range(len(dataset)))
    sampler = SequentialSampler(indices)
    data_loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=1,
        num_workers=4,
        drop_last=False
    )

    Z1 = torch.Tensor(config.N_SAMPLES, config.LATENT_DIM)
    Z2 = torch.Tensor(config.N_SAMPLES, config.LATENT_DIM)

    for i, (x, x_diff) in enumerate(data_loader):

        # Explicitly separate so that this can handle multiple x values without
        # breaking.
        x1 = x[0]
        x2 = x[1]

        x1.requires_grad_()
        x2.requires_grad_()

        x1 = cuda.ize(x1)
        x2 = cuda.ize(x2)

        if mode == 'cca':
            z1, z2 = model.forward([x1, x2])
        elif mode == 'cca_ae':
            (z1, _), z2 = model.forward([x1, x2])
        elif mode == 'cca_vae':
            (z1, _, _, _), z2 = model.forward([x1, x2])
        elif mode == 'cca_2vae':
            (z1, _, _, _), \
            (z2, _, _, _) = model.forward([x1, x2])

        Z1[i] = z1.data
        Z2[i] = z2.data

    torch.save(Z1, '%s/Z1.pt' % directory)
    torch.save(Z2, '%s/Z2.pt' % directory)

# ------------------------------------------------------------------------------

def sbatch_preembed_images(dataset_name, directory):
    """Use the UNIX `sbatch` utility to run a separate program to embed the data
    to avoid memory issues.
    """
    sbatch_contents = gen_sbatch_file(dataset_name, directory)
    sbatch_fname = '%s/sbatch.sh' % directory
    with open(sbatch_fname, 'w+') as f:
        f.write(sbatch_contents)
    cmd = 'sbatch %s' % sbatch_fname
    os.system(cmd)

# ------------------------------------------------------------------------------

def preembed_images(subdir, cfg, model):
    """Embed each image as an average of n_z_per_samp subsamples.
    """
    model.eval()
    model = cuda.ize(model)

    dataset = GTExImages(cfg)
    indices = list(range(len(dataset)))
    data_loader = DataLoader(dataset=dataset,
                             batch_size=1,
                             sampler=SequentialSampler(indices),
                             num_workers=4,
                             pin_memory=cuda.available())

    Z = torch.Tensor(cfg.N_SAMPLES, cfg.LATENT_DIM)
    for i, x in enumerate(data_loader):
        print('Embedded %s-th image.' % i)
        Z[i] = embed_one_image(x, model, dataset.subsample, cfg)
        gc.collect()

    dir_ = '%s/embedded_images.pt' % subdir
    torch.save(Z, dir_)

# ------------------------------------------------------------------------------

def embed_one_image(x, model, subsample, cfg):
    """Embed one image. Wrapping this in a function so that memory for the
    images can be released.
    """
    # Generate a batch of images. Each time we access dataset[i], we
    # subsample the i-th image anew.
    Z = torch.Tensor(cfg.N_Z_PER_SAMPLE, cfg.LATENT_DIM)
    x = x.squeeze(0)

    for i in range(cfg.N_Z_PER_SAMPLE):
        # Subsample and convert to batch of size 1.
        xi          = subsample(x).unsqueeze(0)
        xi          = Variable(xi, requires_grad=False)
        zi, _, _, _ = model.forward(cuda.ize(xi))
        Z[i]        = zi.data

    # The i-th embedding is an average of n_z_per_samp embeddings.
    Zi = Z.mean(dim=0)
    assert Zi.shape == torch.Size([cfg.LATENT_DIM])
    return Zi

# ------------------------------------------------------------------------------

def gen_sbatch_file(dataset_name, directory):
    """Return contents of sbatch file based on experimental setup.
    """
    logfile = '%s/out_emb.txt' % directory

    header = """#!/bin/bash

#SBATCH --mem 250G
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks-per-socket=2
#SBATCH --gres=gpu:1
#SBATCH -o %s
#SBATCH -t 24:00:00
#SBATCH --mail-user=ggundersen@princeton.edu

module load cudatoolkit/8.0 cudann/cuda-8.0/5.1
module load anaconda3
source activate dmcm

cd /scratch/gpfs/gwg3/dmcm\n
""" % logfile

    cmd = 'python embed.py --pre=True --dataset=%s --directory=%s' % (
        dataset_name, directory)
    return header + cmd
