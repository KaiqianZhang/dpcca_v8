"""=============================================================================
Full DMCM model for images and associated high-dimensional signal.

See ParallelTable implementation for analog model:

    https://github.com/amdegroot/pytorch-containers/blob/master/README.md
============================================================================="""

from torch import nn

# ------------------------------------------------------------------------------

class DMCM(nn.Module):

    def __init__(self, mode, cfg):
        """Initialize model for Deep Multimodal Correlation Maximization.
        """
        super(DMCM, self).__init__()

        self.conv_net   = cfg.get_image_net(mode)
        self.sparse_net = cfg.get_genes_net(mode)

        # Matrix network does not need weight initialization because there can
        # be no vanishing gradients.
        self.conv_net.apply(_init_weights_xavier)

# ------------------------------------------------------------------------------

    def forward(self, x):
        """Perform forward pass of images and associated signal through model.
        Output embeddings y1, y2.
        """
        x1, x2 = x
        y1 = self.conv_net.forward(x1)
        y2 = self.sparse_net.forward(x2)
        return y1, y2

# ------------------------------------------------------------------------------

def _init_weights_xavier(m):
    """Credit: https://discuss.pytorch.org/t/weight-initilzation/157/9.
    """
    if isinstance(m, nn.Conv2d):
        # Use Xavier normalization which helps with vanishing gradients:
        #
        #     http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
        #
        # We exclude bias nodes because they are constants and therefore have
        # zero variance.
        nn.init.xavier_uniform(m.weight.data)
