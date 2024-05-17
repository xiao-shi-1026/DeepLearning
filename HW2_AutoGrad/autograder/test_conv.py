import numpy as np
import torch
import sys

# sys.path.append(sys.'./..')
sys.path.append('..')

from mytorch import autograd_engine
from mytorch.functional_hw1 import *
from mytorch.functional_hw2 import *
from mytorch.nn.conv import Conv1d, Conv2d
from helpers import *



def test_cnn1d_layer_forward():
    np.random.seed(0)
    x = np.random.random((1, 3, 5))

    autograd = autograd_engine.Autograd()
    cnn = Conv1d(in_channel=3,
                 out_channel=5,
                 kernel_size=3,
                 downsampling_factor=2,
                 autograd_engine=autograd)
    cnn_out = cnn(x)

    torch_cnn = torch.nn.Conv1d(3, 5, 3, 2)
    torch_cnn.weight = torch.nn.Parameter(
        torch.DoubleTensor(cnn.conv1d_stride1.W))
    torch_cnn.bias = torch.nn.Parameter(
        torch.DoubleTensor(cnn.conv1d_stride1.b))
    torch_x = torch.DoubleTensor(x)
    torch_cnn_out = torch_cnn(torch_x)

    compare_np_torch(cnn_out, torch_cnn_out)
    return True


def test_cnn1d_layer_backward():
    np.random.seed(0)
    x = np.random.random((1, 3, 5))

    autograd = autograd_engine.Autograd()
    cnn = Conv1d(in_channel=3,
                 out_channel=5,
                 kernel_size=3,
                 downsampling_factor=2,
                 autograd_engine=autograd)
    cnn_out = cnn(x)
    autograd.backward(np.ones_like(cnn_out))

    torch_cnn = torch.nn.Conv1d(3, 5, 3, 2)
    torch_cnn.weight = torch.nn.Parameter(
        torch.DoubleTensor(cnn.conv1d_stride1.W))
    torch_cnn.bias = torch.nn.Parameter(
        torch.DoubleTensor(cnn.conv1d_stride1.b))
    torch_x = torch.DoubleTensor(x)
    torch_x.requires_grad = True
    torch_cnn_out = torch_cnn(torch_x)
    torch_cnn_out.sum().backward()

    compare_np_torch(cnn.autograd_engine.gradient_buffer.get_param(x),
                     torch_x.grad)
    compare_np_torch(cnn.conv1d_stride1.dW, torch_cnn.weight.grad)
    compare_np_torch(cnn.conv1d_stride1.db, torch_cnn.bias.grad)
    return True


def test_cnn2d_layer_forward():
    np.random.seed(0)
    x = np.random.random((1, 3, 5, 5))

    autograd = autograd_engine.Autograd()
    cnn = Conv2d(in_channel=3,
                 out_channel=5,
                 kernel_size=3,
                 downsampling_factor=2,
                 autograd_engine=autograd)
    cnn_out = cnn(x)

    torch_cnn = torch.nn.Conv2d(3, 5, 3, 2)
    torch_cnn.weight = torch.nn.Parameter(
        torch.DoubleTensor(cnn.conv2d_stride1.W))
    torch_cnn.bias = torch.nn.Parameter(
        torch.DoubleTensor(cnn.conv2d_stride1.b))
    torch_x = torch.DoubleTensor(x)
    torch_cnn_out = torch_cnn(torch_x)
    compare_np_torch(cnn_out, torch_cnn_out)
    return True


def test_cnn2d_layer_backward():
    np.random.seed(0)
    x = np.random.random((1, 3, 5, 5))
    autograd = autograd_engine.Autograd()
    cnn = Conv2d(in_channel=3,
                 out_channel=5,
                 kernel_size=3,
                 downsampling_factor=2,
                 autograd_engine=autograd)
    cnn_out = cnn(x)
    autograd.backward(np.ones_like(cnn_out))

    torch_cnn = torch.nn.Conv2d(in_channels=3,
                                out_channels=5,
                                kernel_size=3,
                                stride=2)
    torch_cnn.weight = torch.nn.Parameter(
        torch.DoubleTensor(cnn.conv2d_stride1.W))
    torch_cnn.bias = torch.nn.Parameter(
        torch.DoubleTensor(cnn.conv2d_stride1.b))
    torch_x = torch.DoubleTensor(x)
    torch_x.requires_grad = True
    torch_cnn_out = torch_cnn(torch_x)
    torch_cnn_out.sum().backward()

    compare_np_torch(cnn.autograd_engine.gradient_buffer.get_param(x),
                     torch_x.grad)
    compare_np_torch(cnn.conv2d_stride1.dW, torch_cnn.weight.grad)
    compare_np_torch(cnn.conv2d_stride1.db, torch_cnn.bias.grad)
    return True
