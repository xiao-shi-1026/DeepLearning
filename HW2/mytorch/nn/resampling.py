import numpy as np


class Upsample1d():

    def __init__(self, upsampling_factor):
        """
        unsampling_factor(k):
            insert k - 1 zeros between two values in the original array
        """

        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        For example, k = 2, so only 1 zero is inserted between the elements.
        """

        W_out = self.upsampling_factor * (A.shape[2] - 1) + 1 # Calculate the upsampling factor

        Z = np.zeros((A.shape[0], A.shape[1], W_out)) # Create a new array filled with zeros

        Z[..., ::self.upsampling_factor] = A # for every k elements in the new array, copy elements from the old array

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        
        Only the dLdZ entries corresponding to the original input data should be kept, 
        and the dLdZ entries corresponding to the padded 0s should have no effect on
        dLdA and hence should be dropped. This function is just a drop function.
        """

        dLdA = dLdZ[..., ::self.upsampling_factor]

        return dLdA


class Downsample1d():
    """
    Store the input size! Consider a array with size 8 and another with size 7
    While k = 2, forward, we get an array with a same size. When backward, we can only generate a
    size 7 array! 
    """

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """
        self.W_in = A.shape[2]
        Z = A[..., ::self.downsampling_factor]

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """

        W_in = self.W_in
        dLdA = np.zeros((dLdZ.shape[0], dLdZ.shape[1], W_in))
        dLdA[..., ::self.downsampling_factor] = dLdZ

        return dLdA


class Upsample2d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """
        output_height = self.upsampling_factor * (A.shape[2] - 1) + 1 # Calculate the upsampling factor
        output_width = self.upsampling_factor * (A.shape[3] - 1) +1

        Z = np.zeros((A.shape[0], A.shape[1], output_height, output_width))

        Z[..., :: self.upsampling_factor, :: self.upsampling_factor] = A

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        dLdA = dLdZ[..., :: self.upsampling_factor, :: self.upsampling_factor]

        return dLdA


class Downsample2d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """
        self.input_height = A.shape[2]
        self.input_width = A.shape[3]
        Z = A[..., :: self.downsampling_factor, :: self.downsampling_factor]  # TODO

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        dLdA = np.zeros((dLdZ.shape[0], dLdZ.shape[1], self.input_height, self.input_width))

        dLdA[..., :: self.downsampling_factor, :: self.downsampling_factor] = dLdZ

        return dLdA
