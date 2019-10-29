
"""=============================================================================
Run sbatch job to train JMVAE.
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

            cargs            = copy(args)
            cargs.lr         = lr
            cargs.latent_dim = latent_dim

            cargs.directory  = gen_subdir(cargs)
            mem = cargs.mem
            args_dict = vars(cargs)
            del args_dict['mem']
            del args_dict['root']
            del args_dict['script']
            del args_dict['mode']
            contents = jobutils.gen_sbatch_file(args.script, args_dict,
                                                mem=mem, wall_time=cargs.wall_time)
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
    path   = '%s/%s' % (args.root, subdir)
    jobutils.mkdir(path)
    return path

# ------------------------------------------------------------------------------

if __name__ == '__main__':
    now_str = jobutils.get_now_str()

    p = argparse.ArgumentParser()

    p.add_argument('--mode',       type=str,   default='jmvae',
                   choices=['pccavae', 'jmvae'],)
    p.add_argument('--wall_time',  type=int,   default=24)
    p.add_argument('--mem',        type=int,   default=50)
    p.add_argument('--root',       required=True,
                   type=lambda s: 'experiments/%s_%s' % (now_str, s))

    p.add_argument('--dataset',    type=str,   default='gtex')
    p.add_argument('--batch_size', type=int,   default=128)
    p.add_argument('--latent_dim', type=int,   default=[2],    nargs='*')
    p.add_argument('--lr',         type=float, default=[1e-4], nargs='*')
    p.add_argument('--n_epochs',   type=int,   default=1000)
    p.add_argument('--seed',       type=int,   default=0)
    p.add_argument('--cv_pct',     type=float, default=0.1)
    # This argument is for training a VAE on the four tissues sub-dataset.
    args = p.parse_args()

    if args.mode == 'pcca':
        args.script = 'trainpccavae.py'
    else:
        args.script = 'trainjmvae.py'

    jobutils.mkdir(args.root)
    main(args)
