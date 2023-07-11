import math
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules import Module
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split


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
        neuron_num: int, 
        dt=0.001, 
        tau=0.025,
        random_seed=42,
    ):
        super().__init__()
        self.neuron_num = neuron_num
        self.dt = dt
        self.tau = tau

        torch.manual_seed(random_seed)

        self.W_ij = Parameter(torch.randn(neuron_num, neuron_num)) # W_ij initialization

        self.x_t = torch.randn(neuron_num)    # x_(t=0) initialization

    def forward(self):
        x_t_1 = (1 - self.dt/self.tau) * self.x_t - self.dt/self.tau * F.tanh(self.W_ij @ self.x_t)
        self.x_t = x_t_1
        return x_t_1
    


def generate_simulation_data(
    neuron_num = 10,
    dt = 0.001,
    tau = 0.025,
    random_seed=42,
    total_time = 100,
    total_data_size = 1000,
    batch_size = 32,
    num_workers: int = 6, 
    shuffle: bool = False,
) -> DataLoader:
    """
    Generate dataset.
    Return dataloaders and ground truth weight matrix.

    Since this is an unsupervised learning problem, we don't need to split the dataset into train and val.
    """

    simulator = data_simulator(neuron_num=neuron_num, dt=dt, tau=tau)

    data = []
    for i  in range(total_data_size):
        one_sample = []
        for t in range(total_time):
            x_t = simulator.forward()
            x_t = x_t.view(-1, 1)
            one_sample.append(x_t)
        one_sample = torch.cat(one_sample, dim=1).view(1, neuron_num, -1)
        data.append(one_sample)

    data = torch.cat(data, dim=0).float()

    print(data.shape)

    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return dataloader, simulator.W_ij