import numpy as np
from mytorch.functional_hw1 import *
from mytorch.functional_hw2 import *

class Upsample2d():
    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        # TODO: Implement forward (you can rely on HW2P1 code)
        raise NotImplementedError

    def backward(self, dLdZ):
        # TODO: Implement backward (you can rely on HW2P1 code)
        raise NotImplementedError


class Downsample2d():
    def __init__(self, downsampling_factor, autograd_engine):
        self.downsampling_factor = downsampling_factor
        self.autograd_engine = autograd_engine

    def forward(self, A):
        # TODO: Implement forward (you can rely on HW2P1 code)
        # TODO: Add operation to autograd_engine
        raise NotImplementedError


class Upsample1d():
    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        # TODO: Implement forward (you can rely on HW2P1 code)
        raise NotImplementedError

    def backward(self, dLdZ):
        # TODO: Implement backward (you can rely on HW2P1 code)
        raise NotImplementedError


class Downsample1d():
    def __init__(self, downsampling_factor, autograd_engine):
        self.downsampling_factor = downsampling_factor
        self.autograd_engine = autograd_engine

    def forward(self, A):
        # TODO: Implement forward (you can rely on HW2P1 code)
        # TODO: Add operation to autograd_engine
        raise NotImplementedError
