# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
from resampling import *


class Conv1d_stride1():
    def __init__(self, in_channels, out_channels, kernel_size,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
            batch_size: how many data points in the batch
            in_channel: for each data point, how many channels are used to describe the data
            input_size: How much "basic units" are in one data point. For example, a pixel is a basic unit in a picture.
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
            output_size: when stride = 1, output_size = input_size - kernel_size + 1
        """
        self.A = A
        Z = np.zeros((self.A.shape[0], self.W.shape[0], self.A.shape[2] - self.W.shape[2] + 1))

        # for every datapoint
        for i in range(Z.shape[2]): # for every slice
            tmp = self.A[..., i:i + self.W.shape[2]] # get the current slice of A
            # tmp (batch_size, input_channel, slice_size(kernel_size)), W (output_channel, input_channel, kernel_size)
            Z[...,i] = np.tensordot(tmp, self.W, axes = ([1, 2],[1, 2])) # Z (batch_size, out_channels, output_size)
        return Z + self.b.reshape(1, self.b.shape[0], 1)

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """

        self.dLdW = np.zeros(self.W.shape)

        # calculate dLdW

        for i in range(self.dLdW.shape[2]):
            tmp = self.A[..., i:i + dLdZ.shape[2]] # tmp (batch_size, input_channel, slice_size(dLdZ output_size))
            self.dLdW[..., i] = np.tensordot(dLdZ, tmp, axes=([0, 2], [0, 2])) # dLdZ(batch_size, out_channels, output_size), dLdW(output_channel, input_channel, kernel_size)

        # calculate dLdb
        # dLdb (output_channel, )
        self.dLdb = np.sum(dLdZ, axis = (0,2))
        
        # calculate dLdA
        dLdA = np.zeros(self.A.shape)
        padded_dLdZ = np.zeros((dLdZ.shape[0], dLdZ.shape[1], dLdA.shape[2] + self.kernel_size - 1))

        padded_dLdZ[..., self.kernel_size - 1: self.kernel_size - 1 + dLdZ.shape[2]] = dLdZ # pad dLdZ, each side for each channel, pad kernel_size - 1 zeros.
        Flipped_W = self.W[..., ::-1]


        for i in range(dLdA.shape[2]):
            tmp = padded_dLdZ[..., i:i + Flipped_W.shape[2]] # tmp (batch_size, output_channel, kernel_size)
            dLdA[..., i] = np.tensordot(tmp, Flipped_W, axes = ([1, 2],[0, 2])) #Flipped_W(output_channel, input_channel, kernel_size)
        return dLdA


class Conv1d():
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding = 0,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names

        self.stride = stride
        self.pad = padding

        # Initialize Conv1d() and Downsample1d() isntance
        self.conv1d_stride1 = Conv1d_stride1(in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn)
        self.downsample1d = Downsample1d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """

        # Pad the input appropriately using np.pad() function
        A = np.pad(A, pad_width = ((0, 0), (0, 0), (self.pad, self.pad)), mode = 'constant', constant_values = 0)

        # Call Conv1d_stride1
        Z = self.conv1d_stride1.forward(A)

        # downsample
        Z = self.downsample1d.forward(Z)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        # Call downsample1d backward
        dLdA = self.downsample1d.backward(dLdZ)

        # Call Conv1d_stride1 backward
        dLdA = self.conv1d_stride1.backward(dLdA)

        # Unpad the gradient
        gradient = dLdA[..., self.pad: dLdA.shape[2] - self.pad]

        return gradient
