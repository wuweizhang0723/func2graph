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
        tau=0.025,    # momentum of the system, the larger tau is the slower the system returns back to equilibrium
        random_seed=42,
        spike_neuron_num=2,
        spike_input=1,
        total_time=30000,
    ):
        super().__init__()
        self.neuron_num = neuron_num
        self.dt = dt
        self.tau = tau

        self.spike_neuron_num = spike_neuron_num
        self.spike_input = spike_input

        torch.manual_seed(random_seed)

        self.W_ij = torch.randn(neuron_num, neuron_num) # W_ij initialization

        self.x_t = torch.randn(neuron_num)    # x_(t=0) initialization

        self.selected_neurons = torch.randint(low=0, high=neuron_num, size=(total_time, 2))

    def forward(self, current_time_step):
        # For each time step, randomly choose 2 neurons to add input=1
        selected = self.selected_neurons[current_time_step]
        I_t = torch.zeros(self.neuron_num)
        I_t[selected] = self.spike_input

        x_t_1 = (1 - self.dt/self.tau) * self.x_t - self.dt/self.tau * F.tanh(self.W_ij @ self.x_t + I_t)
        self.x_t = x_t_1
        return x_t_1   # this is a vector of size neuron_num
    


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
    split_ratio = 0.8,
) -> DataLoader:
    """
    Generate dataset.
    Return dataloaders and ground truth weight matrix.
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
    train_data_size = int(total_data_size * split_ratio)
    train_dataset, val_dataset = random_split(
        dataset,
        [train_data_size, total_data_size - train_data_size],
        generator=torch.Generator().manual_seed(random_seed),
    )

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return train_dataloader, val_dataloader, simulator.W_ij