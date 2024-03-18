import numpy as np
from nn.activation import *


class GRUCell(object):
    """GRU Cell class."""

    def __init__(self, input_size, hidden_size):
        self.d = input_size
        self.h = hidden_size
        h = self.h
        d = self.d
        self.x_t = 0

        self.Wrx = np.random.randn(h, d)
        self.Wzx = np.random.randn(h, d)
        self.Wnx = np.random.randn(h, d)

        self.Wrh = np.random.randn(h, h)
        self.Wzh = np.random.randn(h, h)
        self.Wnh = np.random.randn(h, h)

        self.brx = np.random.randn(h)
        self.bzx = np.random.randn(h)
        self.bnx = np.random.randn(h)

        self.brh = np.random.randn(h)
        self.bzh = np.random.randn(h)
        self.bnh = np.random.randn(h)

        self.dWrx = np.zeros((h, d))
        self.dWzx = np.zeros((h, d))
        self.dWnx = np.zeros((h, d))

        self.dWrh = np.zeros((h, h))
        self.dWzh = np.zeros((h, h))
        self.dWnh = np.zeros((h, h))

        self.dbrx = np.zeros((h))
        self.dbzx = np.zeros((h))
        self.dbnx = np.zeros((h))

        self.dbrh = np.zeros((h))
        self.dbzh = np.zeros((h))
        self.dbnh = np.zeros((h))

        self.r_act = Sigmoid()
        self.z_act = Sigmoid()
        self.h_act = Tanh()

        # Define other variables to store forward results for backward here

    def init_weights(self, Wrx, Wzx, Wnx, Wrh, Wzh, Wnh, brx, bzx, bnx, brh, bzh, bnh):
        self.Wrx = Wrx
        self.Wzx = Wzx
        self.Wnx = Wnx
        self.Wrh = Wrh
        self.Wzh = Wzh
        self.Wnh = Wnh
        self.brx = brx
        self.bzx = bzx
        self.bnx = bnx
        self.brh = brh
        self.bzh = bzh
        self.bnh = bnh

    def __call__(self, x, h_prev_t):
        return self.forward(x, h_prev_t)

    def forward(self, x, h_prev_t):
        """GRU cell forward.

        Input
        -----
        x: (input_dim)
            observation at current time-step.

        h_prev_t: (hidden_dim)
            hidden-state at previous time-step.

        Returns
        -------
        h_t: (hidden_dim)
            hidden state at current time-step.

        """
        self.x = x
        self.hidden = h_prev_t
        # calculate reset gate
        self.r = self.r_act.forward(np.dot(self.x, self.Wrx.T) + self.brx + np.dot(self.hidden, self.Wrh.T) + self.brh)
        # calculate update gate
        self.z = self.z_act.forward(np.dot(self.x, self.Wzx.T) + self.bzx + np.dot(self.hidden, self.Wzh.T) + self.bzh)
        # calculate input
        self.n = self.h_act.forward(np.dot(self.x, self.Wnx.T) + self.bnx + (np.dot(self.hidden, self.Wnh.T) + self.bnh) * self.r)
        # new hidden state at current time-step
        h_t = self.z * self.hidden + (1 - self.z) * self.n

        # Add your code here.
        # Define your variables based on the writeup using the corresponding
        # names below.
        
        assert self.x.shape == (self.d,)
        assert self.hidden.shape == (self.h,)

        assert self.r.shape == (self.h,)
        assert self.z.shape == (self.h,)
        assert self.n.shape == (self.h,)
        assert h_t.shape == (self.h,) # h_t is the final output of you GRU cell.

        return h_t

    def backward(self, delta):
        """GRU cell backward.

        This must calculate the gradients wrt the parameters and return the
        derivative wrt the inputs, xt and ht, to the cell.

        Input
        -----
        delta: (hidden_dim)
                summation of derivative wrt loss from next layer at
                the same time-step and derivative wrt loss from same layer at
                next time-step.

        Returns
        -------
        dx: (input_dim)
            derivative of the loss wrt the input x.

        dh_prev_t: (hidden_dim)
            derivative of the loss wrt the input hidden h.

        """

        # SOME TIPS:
        # 1) Make sure the shapes of the calculated dWs and dbs match the initalized shapes of the respective Ws and bs
        # 2) When in doubt about shapes, please refer to the table in the writeup.
        # 3) Know that the autograder grades the gradients in a certain order, and the local autograder will tell you which gradient you are currently failing.
        
        # For new hidden state
        dz = delta * (self.hidden - self.n)
        dn = delta * (1 - self.z)
       
        # For input calculation
        self.dWnx = np.outer(self.h_act.backward(dn, self.n).T, self.x)
        self.dbnx = self.h_act.backward(dn, self.n)
        dr = self.h_act.backward(dn, self.n) * (np.dot(self.hidden, self.Wnh.T) + self.bnh)
        self.dWnh = np.outer(self.h_act.backward(dn, self.n) * self.r, self.hidden)
        self.dbnh = self.h_act.backward(dn, self.n) * self.r
        
        # For Update Gate
        self.dWzx = np.outer(self.z_act.backward(dz).T, self.x)
        self.dbzx = self.z_act.backward(dz)
        self.dWzh = np.outer(self.z_act.backward(dz).T, self.hidden)
        self.dbzh = self.z_act.backward(dz)
        
        # For Reset Gate
        self.dWrx = np.outer(self.r_act.backward(dr).T, self.x)
        self.dbrx = self.r_act.backward(dr)
        self.dWrh = np.outer(self.r_act.backward(dr).T, self.hidden)
        self.dbrh = self.r_act.backward(dr)
        
        # dx
        dndx = self.h_act.backward(dn, self.n) @ self.Wnx
        dzdx = self.z_act.backward(dz) @ self.Wzx
        drdx = self.r_act.backward(dr) @ self.Wrx

        dx = dndx + dzdx + drdx

        # dh
        dh_hprev = self.z
        # Notice: if (A @ B)*C, dB = C.T * A 
        dndh = self.r * self.h_act.backward(dn, self.n) @ self.Wnh
        dzdh = self.z_act.backward(dz) @ self.Wzh
        drdh = self.r_act.backward(dr) @ self.Wrh
        
        dh_prev_t = delta * dh_hprev + dndh + dzdh + drdh

        assert dx.shape == (self.d,)
        assert dh_prev_t.shape == (self.h,)

        return dx, dh_prev_t

