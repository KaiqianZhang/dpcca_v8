"""=============================================================================
Models module interface.
============================================================================="""

import torch

from   models.aelinear        import AELinear
from   models.aesigmoid       import AESigmoid
from   models.aesemilinear    import AESemiLinear
from   models.vaesigmoid      import VAESigmoid
from   models.aetanh          import AETanH
from   models.vaetanh         import VAETanH
from   models.alexnetae       import AlexNetAE
from   models.alexnetaebn     import AlexNetAEBN
from   models.alexnetvae      import AlexNetVAE
from   models.alexnetvaebn    import AlexNetVAEBN
from   models.cae             import CAE
from   models.celebaae        import CelebAAE
from   models.celebaae28      import CelebAAE28
from   models.crossmodalityae import CrossModalityAE
from   models.dcca            import DCCA
from   models.dcganae         import DCGANAE
from   models.dcganae128      import DCGANAE128
from   models.dcganvae128     import DCGANVAE128
from   models.dvcca           import DVCCA
from   models.dmcm            import DMCM
from   models.geneae          import GeneAE
from   models.identity        import Identity
from   models.jmvae           import JMVAE
from   models.lenet5ae        import LeNet5AE
from   models.linear          import Linear
from   models.mae             import MAE
from   models.pcca            import PCCA
from   models.pccaopt         import PCCAOpt
from   models.pccavec         import PCCAVec
from   models.pccasimple      import PCCASimple
from   models.dpcca           import DPCCA
from   models.vae             import VAE
from   models.vaelinear       import VAELinear

# ------------------------------------------------------------------------------

def load_trained_model(cfg, ModelCtr, fname, cpu=False):
    """Load trained CAE model based on filename and GPU-availability.
    """
    if cpu:
        state_dict = torch.load(fname, map_location={'cuda:0': 'cpu'})
    else:
        state_dict = torch.load(fname)
    model = ModelCtr(cfg)
    model.load_state_dict(state_dict)
    return model
