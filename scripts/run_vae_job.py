"""=============================================================================
Run sbatch job to train VAE.
============================================================================="""

import argparse
from   copy import copy

import jobutils

# ------------------------------------------------------------------------------

def main(args):
    """Run jobs for entire input parameter space.
    """
    for lr in args.lr:
        for latent_dim in args.latent_dim:
            for beta in args.beta:

                cargs            = copy(args)
                cargs.lr         = lr
                cargs.latent_dim = latent_dim
                cargs.beta       = beta

                cargs.directory  = gen_subdir(cargs)
                mem    = cargs.mem
                n_gpus = cargs.n_gpus
                args_dict = vars(cargs)
                del args_dict['mem']
                del args_dict['root_dir']
                contents = jobutils.gen_sbatch_file(args.script,
                                                    args_dict,
                                                    mem=mem,
                                                    wall_time=cargs.wall_time,
                                                    n_gpus=n_gpus)
                jobutils.run_job(cargs.directory, contents)

# ------------------------------------------------------------------------------

def gen_subdir(args):
    """Return subdirectory name based on experimental setup.
    """
    # Create subdirectory based on experimental setup.
    subdir = args.dataset
    subdir = jobutils.add_desc_to_dir(subdir, True, 'dim', args.latent_dim)
    subdir = jobutils.add_desc_to_dir(subdir, True, 'lr', args.lr)
    subdir = jobutils.add_desc_to_dir(subdir, True, 'epochs', args.n_epochs)
    subdir = jobutils.add_desc_to_dir(subdir, args.beta > 1, 'beta', args.beta)
    subdir = jobutils.add_desc_to_dir(subdir, args.tissue != '', 'tissue',
                                      args.tissue)
    path   = '%s/%s' % (args.root_dir, subdir)
    jobutils.mkdir(path)
    return path

# ------------------------------------------------------------------------------

if __name__ == '__main__':
    p = argparse.ArgumentParser()

    p.add_argument('--mode',       type=str,   default='jmvae',
                   choices=['pccavae', 'jmvae'])
    p.add_argument('--script',     type=str,   default='trainvae.py')
    p.add_argument('--wall_time',  type=int,   default=24)
    p.add_argument('--mem',        type=int,   default=150)

    p.add_argument('--dataset',    type=str,   default='gtex')
    p.add_argument('--batch_size', type=int,   default=128)
    p.add_argument('--n_gpus',     type=int,   default=1)
    p.add_argument('--latent_dim', type=int,   default=[2],    nargs='*')
    p.add_argument('--lr',         type=float, default=[1e-3], nargs='*')
    p.add_argument('--n_epochs',   type=int,   default=100)
    p.add_argument('--seed',       type=int,   default=0)
    p.add_argument('--cv_pct',     type=float, default=0.1)
    # See beta-VAE: https://openreview.net/pdf?id=Sy2fzU9gl.
    p.add_argument('--beta',       type=float, default=[1.0],  nargs='*')
    p.add_argument('--modality',   type=str,   default='images')
    p.add_argument('--tissue',     type=str,   default='')
    args = p.parse_args()

    dir_ = 'ivae' if args.modality == 'images' else 'gvae'
    args.root_dir = 'experiments/%s' % dir_
    main(args)
