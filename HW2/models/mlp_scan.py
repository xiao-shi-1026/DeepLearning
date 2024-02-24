# DO NOT import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

from flatten import *
from Conv1d import *
from linear import *
from activation import *
from loss import *
import numpy as np
import os
import sys

sys.path.append('mytorch')


class CNN_SimpleScanningMLP():
    def __init__(self):
        # Your code goes here -->
        # self.conv1 = ???
        # self.conv2 = ???
        # self.conv3 = ???
        # ...
        # <---------------------
        # in_channels, out_channels, kernel_size, stride, padding = 0, weight_init_fn=None, bias_init_fn=None
        self.conv1 = Conv1d(24, 8, 8, 1)
        self.conv2 = Conv1d(8, 16, 1, 1)
        self.conv3 = Conv1d(16, 4, 1, 4)
        self.layers = [self.conv1, ReLU(), self.conv2, ReLU(), self.conv3, Flatten()] # TODO: Add the layers in the correct order

    def init_weights(self, weights):
        # Load the weights for your CNN from the MLP Weights given
        # w1, w2, w3 contain the weights for the three layers of the MLP
        # Load them appropriately into the CNN
        # W (output_channel, input_channel, kernel_size)
        w1, w2, w3 = weights

        self.conv1.conv1d_stride1.W = w1.reshape(8, 24, 8).transpose(2,1,0)
        self.conv2.conv1d_stride1.W = w2.reshape(1, 8, 16).transpose(2,1,0)
        self.conv3.conv1d_stride1.W = w3.reshape(1, 16, 4).transpose(2,1,0)

    def forward(self, A):
        """
        Do not modify this method

        Argument:
            A (np.array): (batch size, in channel, in width)
        Return:
            Z (np.array): (batch size, out channel , out width)
        """

        Z = A
        for layer in self.layers:
            Z = layer.forward(Z)
        return Z

    def backward(self, dLdZ):
        """
        Do not modify this method

        Argument:
            dLdZ (np.array): (batch size, out channel, out width)
        Return:
            dLdA (np.array): (batch size, in channel, in width)
        """

        for layer in self.layers[::-1]:
            dLdA = layer.backward(dLdA)
        return dLdA


class CNN_DistributedScanningMLP():
    def __init__(self):
        # Your code goes here -->
        # self.conv1 = ???
        # self.conv2 = ???
        # self.conv3 = ???
        # ...
        # <---------------------
        # in_channels, out_channels, kernel_size, stride, padding = 0, weight_init_fn=None, bias_init_fn=None
        self.conv1 = Conv1d(24, 2, 2, 2)
        self.conv2 = Conv1d(2, 8, 2, 2)
        self.conv3 = Conv1d(8, 4, 2, 1)
        self.layers = [self.conv1, ReLU(), self.conv2, ReLU(), self.conv3, Flatten()] # TODO: Add the layers in the correct order

    def __call__(self, A):
        # Do not modify this method
        return self.forward(A)

    def init_weights(self, weights):
        # Load the weights for your CNN from the MLP Weights given
        # w1, w2, w3 contain the weights for the three layers of the MLP
        # Load them appropriately into the CNN
        # W (output_channel, input_channel, kernel_size)
        # (batch_size, in_channels, input_size)
        w1, w2, w3 = weights

        self.conv1.conv1d_stride1.W = w1[0:48, 0:2].reshape(2, 24, 2).transpose(2,1,0)
        self.conv2.conv1d_stride1.W = w2[0:4, 0:8].reshape(2, 2, 8).transpose(2,1,0)
        self.conv3.conv1d_stride1.W = w3.reshape(2, 8, 4).transpose(2,1,0)
    def forward(self, A):
        """
        Do not modify this method

        Argument:
            A (np.array): (batch size, in channel, in width)
        Return:
            Z (np.array): (batch size, out channel , out width)
        """

        Z = A
        for layer in self.layers:
            Z = layer.forward(Z)
        return Z

    def backward(self, dLdZ):
        """
        Do not modify this method

        Argument:
            dLdZ (np.array): (batch size, out channel, out width)
        Return:
            dLdA (np.array): (batch size, in channel, in width)
        """
        dLdA = dLdZ
        for layer in self.layers[::-1]:
            dLdA = layer.backward(dLdA)
        return dLdA
