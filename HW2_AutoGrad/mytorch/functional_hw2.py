import numpy as np
from mytorch.autograd_engine import Autograd


def conv1d_stride1_backward(dLdZ, A, weight, bias):
    """
    Inputs
    ------
    dLdz:   Gradient from next layer
    A:      Input
    weight: Model param
    bias:   Model param

    Returns
    -------
    dLdA, dLdW, dLdb
    """
    # NOTE: You can use code from HW2P1!
    raise NotImplementedError


def conv2d_stride1_backward(dLdZ, A, weight, bias):
    """
    Inputs
    ------
    dLdz:   Gradient from next layer
    A:      Input
    weight: Model param
    bias:   Model param

    Returns
    -------
    dLdA, dLdW, dLdb
    """
    # NOTE: You can use code from HW2P1!
    raise NotImplementedError


def downsampling1d_backward(dLdZ, A, downsampling_factor):
    """
    Inputs
    ------
    dLdz:                   Gradient from next layer
    A:                      Input
    downsampling_factor:    NOTE: for the gradient buffer to work, 
                            this has to be a np.array. 

    Returns
    -------
    dLdA, dLdW, dLdb
    """
    # NOTE: You can use code from HW2P1!
    raise NotImplementedError


def downsampling2d_backward(dLdZ, A, downsampling_factor):
    """
    Inputs
    ------
    dLdz:                   Gradient from next layer
    A:                      Input
    downsampling_factor:    NOTE: for the gradient buffer to work, 
                            this has to be a np.array. 

    Returns
    -------
    dLdA, dLdW, dLdb
    """
    # NOTE: You can use code from HW2P1!
    raise NotImplementedError


def flatten_backward(dLdZ, A):
    """
    Inputs
    ------
    dLdz:   Gradient from next layer
    A:      Input

    Returns
    -------
    dLdA
    """
    # NOTE: You can use code from HW2P1!
    raise NotImplementedError
