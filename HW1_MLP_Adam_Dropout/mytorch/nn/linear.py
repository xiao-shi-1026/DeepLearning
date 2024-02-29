import numpy as np


class Linear:

    def __init__(self, in_features, out_features, debug=False):
        """
        Initialize the weights and biases with zeros
        Checkout np.zeros function.
        Read the writeup to identify the right shapes for all.
        """
        self.W = np.zeros((out_features, in_features))  # TODO
        self.b = np.zeros((out_features, 1))  # TODO

        self.debug = debug

    def forward(self, A):
        """
        :param A: Input to the linear layer with shape (N, C0)
        :return: Output Z of linear layer with shape (N, C1)
        Read the writeup for implementation details
        """
        self.A = A
        self.N = A.shape[0]  # store the batch size of input
        # Think how will self.Ones helps in the calculations and uncomment below
        self.Ones = np.ones((self.N, 1))
        self.Z = self.A @ self.W.T + self.Ones * self.b.T

        return self.Z

    def backward(self, dLdZ) -> np.array:
        """
        :param dLdZ: The e derivative of the loss with respect to Y, shape (N, C0)
        """
        dLdA = dLdZ @ self.W
        self.dLdW = dLdZ.T @ self.A
        self.dLdb = dLdZ.T @ self.Ones

        if self.debug:
            
            self.dLdA = dLdA
        
        return dLdA
