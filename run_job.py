"""=============================================================================
Script for training DMCM on Tiger GPU.
============================================================================="""

import argparse
from   copy import copy
import itertools
import sys
import jobutils


# ------------------------------------------------------------------------------

def main(args):
    """Run jobs for entire input parameter space.
    """
    iterables   = []
    iter_fields = []
    for f in vars(args):
        a = getattr(args, f)
        if type(a) is list:
            iterables.append(a)
            iter_fields.append(f)

    for a in itertools.product(*iterables):
        for f in iter_fields:
            setattr(args, f, None)
        cargs = copy(args)
        for i, f in enumerate(iter_fields):
            setattr(cargs, f, a[i])

        cargs.directory  = gen_subdir(cargs, iter_fields)

        mem    = cargs.mem
        n_gpus = cargs.n_gpus
        args_dict = vars(cargs)
        del args_dict['mem']
        del args_dict['root']
        del args_dict['exec_cmd']

        contents = jobutils.gen_sbatch_file(args.exec_cmd,
                                            args.script,
                                            args_dict,
                                            mem=mem,
                                            wall_time=cargs.wall_time,
                                            n_gpus=n_gpus)

        jobutils.run_job(cargs.directory, contents)


# ------------------------------------------------------------------------------

def gen_subdir(cargs, iter_fields):
    """Return subdirectory name based on experimental setup.
    """
    # Create root directory.
    jobutils.mkdir(cargs.root)

    subdir = '%s/%s' % (args.root, args.dataset)
    subdir = jobutils.add_desc_to_dir(subdir, True, 'mode', args.mode)

    for f in iter_fields:
        subdir = jobutils.add_desc_to_dir(subdir, True, f, getattr(cargs, f))

    jobutils.mkdir(subdir)
    return subdir


# ------------------------------------------------------------------------------

if __name__ == '__main__':
    now_str = jobutils.get_now_str()
    p = argparse.ArgumentParser()

    p.add_argument('--root',
                   required=True,
                   type=lambda s: 'experiments/%s_%s' % (now_str, s))

    p.add_argument('--n_epochs',   type=int,   default=1000)
    p.add_argument('--batch_size', type=int,   default=128)
    p.add_argument('--wall_time',  type=int,   default=24)
    p.add_argument('--cv_pct',     type=float, default=0.1)
    p.add_argument('--mem',        type=int,   default=150)
    p.add_argument('--n_gpus',     type=int,   default=1)
    p.add_argument('--seed',       type=int,   default=[0],    nargs='*')

    p.add_argument('--dataset',    type=str,   default='gtexv8')
    p.add_argument('--latent_dim', type=int,   default=[100],  nargs='*')
    p.add_argument('--lr',         type=float, default=[1e-3], nargs='*')
    p.add_argument('--l1_coef',    type=float, default=[0.0],  nargs='*')
    p.add_argument('--em_iters',   type=int,   default=[1],    nargs='*')
    p.add_argument('--clip',       type=float, default=1)

    args = p.parse_args()

    # Cache the command used to run this script. This way experiments are more
    # reproducible.
    args.exec_cmd = " ".join(sys.argv[:])
    args.script = 'traindpcca.py'

    main(args)
