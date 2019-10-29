"""=============================================================================
Functions for normalizing images and vectors for neural networks.
============================================================================="""

def normalize_inputs(X):
    """For why, see:

        https://www.coursera.org/learn/deep-neural-network/lecture/lXv6U/
            normalizing-inputs

    For how, see:

        http://cs231n.github.io/neural-networks-2/#datapre
    """
    # Perform mean subtraction.
    X -= X.mean(dim=0)
    # Standardize data to be approximately same scale.
    X /= X.std(dim=0)
    return X
