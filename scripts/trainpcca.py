"""=============================================================================
Train probabilistic CCA.
============================================================================="""

import argparse
import time
import os

if '/scratch/gpfs/gwg3/dmcm' in os.path.dirname(os.path.realpath(__file__)):
    import matplotlib
    matplotlib.use('agg')

import matplotlib.pyplot as plt
import torch
import torch.utils.data
from   torchvision.utils import save_image

import cuda
from   data import loader
from   models import PCCA
import pprint

# ------------------------------------------------------------------------------

device = cuda.device()

# ------------------------------------------------------------------------------

def main(args):
    """Main program: train -> test once per epoch while saving samples as
    needed.
    """
    start_time = time.time()
    pprint.set_logfiles(args.directory)

    cfg = loader.get_config(args.dataset)
    train_loader, _ = loader.get_data_loaders(cfg, 1, 0, True, 0.1)
    n  = len(train_loader.sampler.indices)
    y1 = torch.empty(n, cfg.N_CHANNELS, cfg.IMG_SIZE, cfg.IMG_SIZE)
    y2 = torch.empty(n, cfg.N_GENES)
    for i, (y1i, y2i) in enumerate(train_loader):
        y1[i] = y1i.squeeze(0)
        y2[i] = y2i
    y = torch.cat([y1.view(-1, cfg.N_PIXELS), y2], dim=1)
    y = y.to(device)
    pprint.log('Data loaded.')

    model = PCCA(latent_dim=args.latent_dim,
                 dims=[cfg.N_PIXELS, cfg.N_GENES],
                 gene_low_rank=cfg.GENE_EMBED_DIM,
                 max_iters=args.em_iters,
                 differentiable=False,
                 debug=True)
    model = model.to(device)
    pprint.log('Model instantiated.')

    model.em_bishop(y.t())
    z = model.estimate_z_given_y(y.t(), threshold=None)

    Lambda, Psi_diag = model.tile_params()
    yr = Lambda @ z

    print(model.nlls)

    nc  = cfg.N_CHANNELS
    w   = cfg.IMG_SIZE
    y1r = yr[:, :cfg.N_PIXELS].view(-1, nc, w, w)
    y2r = yr[:, cfg.N_PIXELS:]
    pprint.log('Model fit.')
    pprint.log('Z estimated.')

    for i, img in enumerate(y1r):
        if i > 64:
            break
        fname = '%s/sample_images_%s.png' % (args.directory, i)
        img = img.view(-1, cfg.N_CHANNELS, cfg.IMG_SIZE, cfg.IMG_SIZE)
        save_image(img.detach().cpu(), fname)

    for i, genes in enumerate(y2r):
        genes = genes.detach().cpu()
        if i > 1000:
            break
        plt.scatter(genes[0], genes[1])
    plt.savefig('%s/sample_genes.png' % args.directory)

    hours = round((time.time() - start_time) / 3600, 1)
    pprint.log_section('Job complete in %s hrs.' % hours)

    # filename = '%s/model.joblib.pkl' % args.directory
    # joblib.dump(model, filename, compress=9)
    filename = '%s/model.pt' % args.directory
    state = model.state_dict()
    torch.save(state, filename)
    pprint.log_section('Model saved.')

# ------------------------------------------------------------------------------

if __name__ == '__main__':
    p = argparse.ArgumentParser()

    p.add_argument('--directory',  type=str,   default='experiments/test')
    p.add_argument('--wall_time',  type=int,   default=24)
    p.add_argument('--seed',       type=int,   default=0)
    p.add_argument('--dataset',    type=str,   default='mnist')
    p.add_argument('--latent_dim', type=int,   default=10)
    p.add_argument('--em_iters',   type=int,   default=100)

    args, _ = p.parse_known_args()

    torch.manual_seed(args.seed)
    main(args)
