import math
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules import Module
import numpy as np

class data_simulator(Module):
    """
    data simulator for neuron activity
    Formula:
        x_i(t+1) = (1 - dt/tau) * x_i(t) - dt/tau * tanh{sum_j[W_ij * x_j(t) + I_i(t)]}

        x_i(t=0) is sampled from standard normal distribution
        W_ij can be sampled from 1) standard normal distribution 2) cluster 3) nearest neighbor
    """
    def __init__(
        self, 
        n, 
        dt=0.001, 
        tau=0.025,
    ):
        super(data_simulator, self).__init__()
        self.n = Parameter(n)
        self.dt = Parameter(dt)
        self.tau = Parameter(tau)

        self.W_ij = Parameter(torch.randn(n, n)) # W_ij initialization

        self.x_t = torch.randn(n)    # x_(t=0) initialization

    def forward(self):
        x_t_1 = (1 - self.dt/self.tau) * self.x_t - self.dt/self.tau * F.tanh(self.W_ij @ self.x_t)
        self.x_t = x_t_1
        return x_t_1