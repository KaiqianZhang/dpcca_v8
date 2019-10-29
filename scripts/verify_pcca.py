"""=============================================================================
Verify refactored PCCA using Woodbury identity works as expected.
============================================================================="""

import matplotlib.pyplot as plt
import torch
from   models import PCCASimple, PCCAVec, PCCAOpt, PCCA

from   tests.utils import gen_simple_dataset

# ------------------------------------------------------------------------------

def load_and_visualize_data(p1, p2, k, n, show):
    y1, y2, z = gen_simple_dataset(p1, p2, k, n, 1.0, 10)

    y1 = y1 - y1.mean(dim=0)
    y2 = y2 - y2.mean(dim=0)
    y = torch.cat([y1, y2], dim=1)
    y = y.t()

    if show:
        fig, ax = plt.subplots()
        ax.scatter(y1[:, 0], y1[:, 1], c='red', marker='.')
        ax.scatter(y2[:, 0], y2[:, 1], c='blue', marker='.')
        plt.show()

    return y1, y2, y

# ------------------------------------------------------------------------------

if __name__ == '__main__':
    k = 10
    n = 1000
    p1 = 50
    p2 = 30
    y1, y2, y = load_and_visualize_data(p1, p2, k, n, False)

    kwargs = {
        'latent_dim': k,
        'dims': [p1, p2],
        'n_iters': 10,
        'private_z': True
    }

    for ModelCtr in [PCCA]:#[PCCASimple, PCCAVec, PCCAOpt]:
        pcca = ModelCtr(**kwargs)

        pcca.forward(y)
        print('Model fitted')

        x1r, x2r = pcca.sample(y, 500)
        print('Sampling done')

        fig, ax = plt.subplots()
        ax.scatter(y1[:, 0], y1[:, 1], c='red', marker='.')
        ax.scatter(y2[:, 0], y2[:, 1], c='blue', marker='.')
        ax.scatter(x1r[:, 0], x1r[:, 1], c='orange', marker='*')
        ax.scatter(x2r[:, 0], x2r[:, 1], c='cyan', marker='*')
        plt.show()

        plt.scatter(list(range(len(pcca.nlls))), pcca.nlls)
        plt.show()
