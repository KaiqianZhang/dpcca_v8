"""=============================================================================
General utility functions.
============================================================================="""

import datetime
import os

# ------------------------------------------------------------------------------

def run_job(fpath, sbatch_contents):
    """Create sbatch file and run job.
    """
    sbatch_fname = '%s/sbatch.sh' % fpath
    with open(sbatch_fname, 'w+') as f:
        f.write(sbatch_contents)
    cmd = 'sbatch %s' % sbatch_fname
    os.system(cmd)

# ------------------------------------------------------------------------------

def mkdir(directory):
    """Make directory if necessary.
    """
    if not os.path.isdir(directory):
        os.system('mkdir %s' % directory)

# ------------------------------------------------------------------------------

def sbatch(script, args, mem=50, directory='scratch', wall_time=24):
    """Run sbatch command based on script inputs.
    """
    mkdir(directory)
    sbatch_fname = '%s/sbatch.sh' % directory
    contents = gen_sbatch_file(script, args, mem, directory, wall_time)
    with open(sbatch_fname, 'w+') as f:
        f.write(contents)
    cmd = 'sbatch %s' % sbatch_fname
    os.system(cmd)

# ------------------------------------------------------------------------------

def gen_sbatch_file(run_job_cmd, script, args, mem=50, wall_time=24, n_gpus=1):
    """Return contents of sbatch file based on experimental setup.
    """
    logfile = '%s/out.txt' % args['directory']
    if type(mem) is int:
        mem = str(mem) + 'G'
    # The -O flag means "optimized". Also disables assert statements.
    script_cmd = 'python -O %s %s' % (script, args_to_cmds(args))

    header = """#!/bin/bash

# The Python script and arguments used to generate this file:
#
# python %s

#SBATCH --mem=%s
#SBATCH --nodes=1
#SBATCH --gres=gpu:%s
#SBATCH --ntasks-per-node=5
#SBATCH --output=%s
#SBATCH --time=%s:00:00

module load cudatoolkit/8.0 cudann/cuda-8.0/5.1
module load anaconda3
source activate dmcm

cd /scratch/gpfs/gwg3/dpcca_v8\n
""" % (run_job_cmd, mem, n_gpus, logfile, wall_time)
    return header + script_cmd

# ------------------------------------------------------------------------------

def args_to_cmds(args):
    """Convert args dictionary to command line arguments. For example:

    >>> args_to_cmds({ 'foo': 'bar', 'qux': 'baz' })
    '--foo=bar --qux=baz'
    """
    result = ''
    for key, val in args.items():
        result += '--%s=%s ' % (key, val)
    return result

# ------------------------------------------------------------------------------

def get_n_params(model):
    """Return number of model parameters.
    """
    total = 0
    for param in model.parameters():
        n = 1
        for s in param.size():
            n = n * s
        total += n
    return total

# ------------------------------------------------------------------------------

def model_size_in_GB(model):
    """Return model size in gigabytes.
    """
    n_params = get_n_params(model)
    bytes = (n_params * 4)
    gbytes = bytes / 1024**3
    return gbytes

# ------------------------------------------------------------------------------

def add_desc_to_dir(directory, condition, key, value):
    """Return `directory` with description (`key`, `value`) pair if `condition`
    is met, else return `directory` unmodified.
    """
    if condition:
        directory = '%s_%s-%s' % (directory, key, value)
    return directory

# ------------------------------------------------------------------------------

def add_arg_to_cmd(cmd, condition, key, value):
    """Return `cmd` with additional argument if `condition` is met, else return
    `cmd` unmodified.
    """
    if condition:
        cmd += ' --%s=%s' % (key, value)
    return cmd

# ------------------------------------------------------------------------------

def get_now_str():
    """Return date string for experiments folders, e.g.: '20180621'.
    """
    now     = datetime.datetime.now()
    day     = '0%s' % now.day if now.day < 10 else now.day
    month   = '0%s' % now.month if now.month < 10 else now.month
    now_str = '%s%s%s' % (now.year, month, day)
    return now_str
