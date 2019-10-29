"""=============================================================================
Manage CUDA-related utility functions.
============================================================================="""

import torch

# ------------------------------------------------------------------------------

def ize(obj):
    """Return CUDA-ized version of object if CUDA is available, else return
    object unchanged.
    """
    return obj.to(device())

# ------------------------------------------------------------------------------

def device():
    """Return current CUDA device if on GPUs else CPU device.
    """
    if torch.cuda.is_available():
        return torch.cuda.current_device()
    else:
        return torch.device('cpu')

# ------------------------------------------------------------------------------

def parallel():
    """Return `True` if we want to run program in parallel, `False` otherwise.
    """
    return torch.cuda.device_count() > 1
