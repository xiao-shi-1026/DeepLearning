import numpy as np


class MSELoss:

    def forward(self, A, Y) -> float:
        """
        Calculate the Mean Squared error
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: MSE Loss(scalar)

        """

        self.A = A
        self.Y = Y
        self.N = A.shape[0]
        self.C = A.shape[1]
        se = (self.A - self.Y) * (self.A - self.Y)
        sse = np.ones(self.N).T @ se @np.ones(self.C)
        mse = sse / (self.N * self.C)

        return mse

    def backward(self) -> np.array:

        dLdA = 2 * (self.A - self.Y) / (self.N * self.C)

        return dLdA


class CrossEntropyLoss:

    def forward(self, A, Y) -> float:
        """
        Calculate the Cross Entropy Loss
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: CrossEntropyLoss(scalar)

        Refer the the writeup to determine the shapes of all the variables.
        Use dtype ='f' whenever initializing with np.zeros()
        """
        self.A = A
        self.Y = Y
        N = self.A.shape[0]  # TODO
        C = self.A.shape[1]  # TODO

        Ones_C = np.ones((C, 1))  # TODO
        Ones_N = np.ones((N, 1))  # TODO

        self.softmax = np.exp(A) / (np.exp(A) @ Ones_C)
        crossentropy = (-self.Y * np.log(self.softmax)) @ Ones_C  # TODO
        sum_crossentropy = Ones_N.T @ crossentropy  # TODO
        L = sum_crossentropy / N

        return L

    def backward(self) -> np.array:

        dLdA = (self.softmax - self.Y) / self.A.shape[0] # TODO

        return dLdA
