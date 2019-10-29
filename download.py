"""=============================================================================
Download experimental directory
============================================================================="""

import argparse
import os

# ------------------------------------------------------------------------------

def mkdir(directory):
    """Make directory if it does not exist. Void return.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

# ------------------------------------------------------------------------------

def download(directory, download_model, loc):

    remote = '%s/%s' % (loc, directory)
    local = '/Users/gwg/dmcm/experiments/'
    mkdir(local)
    if download_model:
        cmd = 'rsync --progress -r ' \
              'gwg3@tigergpu.princeton.edu:%s %s' % (remote, local)
    else:
        cmd = 'rsync --progress -r --exclude=model.pt ' \
              'gwg3@tigergpu.princeton.edu:%s %s' % (remote, local)
    os.system(cmd)

# ------------------------------------------------------------------------------

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--directory', type=str, default='', required=False)
    p.add_argument('--model',     type=int, default=0,  required=False)
    p.add_argument('--archived',  type=int, default=0,  required=False)

    args = p.parse_args()

    args.model    = bool(args.model)
    args.archived = bool(args.archived)

    if args.archived:
        loc = '/tigress/gwg3/dmcm_experiments_backup'
    else:
        loc = '/scratch/gpfs/gwg3/dmcm/experiments'

    download(args.directory, args.model, loc)

