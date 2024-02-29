import numpy as np


class BatchNorm1d:

    def __init__(self, num_features, alpha=0.9):

        self.alpha = alpha
        self.eps = 1e-8

        self.BW = np.ones((1, num_features))
        self.Bb = np.zeros((1, num_features))
        self.dLdBW = np.zeros((1, num_features))
        self.dLdBb = np.zeros((1, num_features))

        # Running mean and variance, updated during training, used during
        # inference
        self.running_M = np.zeros((1, num_features))
        self.running_V = np.ones((1, num_features))

    def forward(self, Z, eval=False):
        """
        The eval parameter is to indicate whether we are in the
        training phase of the problem or the inference phase.
        So see what values you need to recompute when eval is False.
        """
        self.Z = Z
        self.N = self.Z.shape[0]  # batch size
        self.M = np.mean(Z, axis = 0).reshape((1, self.Z.shape[1]))  # mini-batch per feature mean
        self.V = np.var(Z, axis = 0).reshape((1, self.Z.shape[1]))  # mini-batch per feature variance

        if eval == False:
            # training mode
            self.NZ = (self.Z - np.ones((self.Z.shape[0], 1)) @ self.M) / (np.ones((self.Z.shape[0], 1)) @ np.sqrt(self.V + self.eps))
            self.BZ = self.NZ * (np.ones((self.Z.shape[0], 1)) @ self.BW) + np.ones((self.Z.shape[0], 1)) @ self.Bb

            self.running_M = self.alpha * self.running_M + (1 - self.alpha) * self.M
            self.running_V = self.alpha * self.running_V + (1 - self.alpha) * self.V
        else:
            # inference mode
            self.NZ = (self.Z - np.ones((self.Z.shape[0], 1)) @ self.running_M) / (np.ones((self.Z.shape[0], 1)) @ np.sqrt(self.running_V + self.eps))
            self.BZ = self.NZ * (np.ones((self.Z.shape[0], 1)) @ self.BW) + (np.ones((self.Z.shape[0], 1)) @ self.Bb)

        return self.BZ

    def backward(self, dLdBZ):

        self.dLdBW = np.ones((1,self.N)) @ dLdBZ

        self.dLdBb = np.ones((1,self.N)) @ (dLdBZ * self.NZ)

        dLdNZ = dLdBZ * self.BW

        dLdV = -0.5 * (np.ones((1, self.N)) @ (dLdNZ * (self.Z - self.M) * ((self.V + self.eps) ** (-3/2))))
        
        dLdM = np.ones((1, self.N)) @ (dLdNZ * (-(self.V + self.eps) ** (-0.5) - 0.5 * (self.Z - self.M) * ((self.V + self.eps) ** (-3/2)) * (-2 / self.N) * (np.ones((1, self.N)) @ (self.Z - self.M))))

        dLdZ = dLdNZ * ((self.V + self.eps) ** (-1/2)) + dLdV * (2 / self.N) * (self.Z - self.M) + (1 / self.N) * dLdM

        return dLdZ
