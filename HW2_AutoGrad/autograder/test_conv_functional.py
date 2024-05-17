import numpy as np
import sys
sys.path.append('..')

import torch
import torch.nn as nn
import torch.nn.functional as F

from mytorch.functional_hw2 import *

import helpers


def test_conv1d_backward():

    batch_size = 3
    in_channel, out_channel = 2, 10
    input_size, kernel_size = 5, 3
    stride = 1
    output_size = ((input_size - kernel_size) // stride) + 1

    grad_output = np.ones((batch_size, out_channel, output_size))

    # Student implementation
    A_np = np.random.rand(batch_size, in_channel, input_size)
    weight_np = np.random.rand(out_channel, in_channel, kernel_size)
    bias_np = np.zeros(out_channel)

    # Torch implementation
    A_torch = torch.from_numpy(A_np).requires_grad_()
    weight_torch = nn.parameter.Parameter(torch.from_numpy(weight_np))
    bias_torch = nn.parameter.Parameter(torch.from_numpy(bias_np))

    out_torch = F.conv1d(A_torch, weight_torch, bias=bias_torch, 
                         stride=1, padding=0, dilation=1, groups=1)
    loss_torch = out_torch.sum()
    loss_torch.backward()

    dLdA, dLdW, dLdb = conv1d_stride1_backward(
        grad_output, A_np, weight_np, bias_np
    )

    helpers.compare_np_torch(dLdA, A_torch.grad)
    helpers.compare_np_torch(dLdW, weight_torch.grad)
    helpers.compare_np_torch(dLdb, bias_torch.grad)

    return True



def test_conv2d_backward():
    batch_size = 3
    in_channel, out_channel = 5, 10
    input_width, input_height, kernel_size = 5, 5, 3
    stride = 1
    out_width = (input_width - kernel_size) // stride + 1
    out_height = (input_height - kernel_size) // stride + 1
    grad_output = np.ones((batch_size, out_channel, out_width, out_height))
    
    # Student implementation
    A_np = np.random.rand(batch_size, in_channel, input_width, input_height)
    weight_np = np.random.rand(out_channel, in_channel, kernel_size, kernel_size)
    bias_np = np.zeros(out_channel)

    # Torch implementation
    A_torch = torch.from_numpy(A_np).requires_grad_()
    weight_torch = nn.parameter.Parameter(torch.from_numpy(weight_np))
    bias_torch = nn.parameter.Parameter(torch.from_numpy(bias_np))

    out_torch = F.conv2d(A_torch, weight_torch, bias=bias_torch, 
                         stride=1, padding=0, dilation=1, groups=1)
    loss_torch = out_torch.sum()
    loss_torch.backward()

    dLdA, dLdW, dLdb = conv2d_stride1_backward(
        grad_output, A_np, weight_np, bias_np
    )

    helpers.compare_np_torch(dLdA, A_torch.grad)
    helpers.compare_np_torch(dLdW, weight_torch.grad)
    helpers.compare_np_torch(dLdb, bias_torch.grad)

    return True
