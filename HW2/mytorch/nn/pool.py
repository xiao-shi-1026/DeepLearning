import numpy as np
from resampling import *


class MaxPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A

        Z = np.zeros((self.A.shape[0], self.A.shape[1], self.A.shape[2] - self.kernel + 1, self.A.shape[3] - self.kernel + 1))
        
        for i in range(Z.shape[2]):
            for j in range(Z.shape[3]):
                tmp = self.A[..., i:i + self.kernel, j:j + self.kernel]
                Z[..., i, j] = np.max(tmp, axis = (2, 3))
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdA = np.zeros(self.A.shape)
        for i in range(dLdZ.shape[2]):
            for j in range(dLdZ.shape[3]):
                
                tmp = self.A[..., i:i + self.kernel, j:j + self.kernel]
                mask = (tmp == np.max(tmp, axis = (2, 3), keepdims = True))
                dLdA[..., i:i + self.kernel, j:j + self.kernel] += mask * dLdZ[..., i, j, np.newaxis, np.newaxis]
        
        return dLdA

class MeanPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A

        Z = np.zeros((self.A.shape[0], self.A.shape[1], self.A.shape[2] - self.kernel + 1, self.A.shape[3] - self.kernel + 1))
        
        for i in range(Z.shape[2]):
            for j in range(Z.shape[3]):
                tmp = self.A[..., i:i + self.kernel, j:j + self.kernel]
                Z[..., i, j] = np.mean(tmp, axis = (2, 3))
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        dLdA = np.zeros(self.A.shape)
        for i in range(dLdZ.shape[2]):
            for j in range(dLdZ.shape[3]):
                tmp = self.A[..., i:i + self.kernel, j:j + self.kernel]
                mask = np.ones_like(tmp) / (self.kernel * self.kernel)
                dLdA[..., i:i + self.kernel, j:j + self.kernel] += mask * dLdZ[..., i, j, np.newaxis, np.newaxis]

        return dLdA


class MaxPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.maxpool2d_stride1 = MaxPool2d_stride1(kernel)
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        A = self.maxpool2d_stride1.forward(A)
        Z = self.downsample2d.forward(A)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        tmp = self.downsample2d.backward(dLdZ)
        dLdA = self.maxpool2d_stride1.backward(tmp)
        
        return dLdA


class MeanPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.meanpool2d_stride1 = MeanPool2d_stride1(kernel)  # TODO
        self.downsample2d = Downsample2d(stride)  # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        A = self.meanpool2d_stride1.forward(A)
        Z = self.downsample2d.forward(A)
        
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        tmp = self.downsample2d.backward(dLdZ)
        dLdA = self.meanpool2d_stride1.backward(tmp)

        return dLdA
