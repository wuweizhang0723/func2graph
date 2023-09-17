import math
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn.modules import Module
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split
from os.path import dirname, join as pjoin
import scipy.io as sio
import pandas as pd

from func2graph import tools


class data_simulator(Module):
    """
    data simulator for neuron activity
    Formula:
        x_i(t+1) = (1 - dt/tau) * x_i(t) - dt/tau * tanh{sum_j[W_ij * x_j(t) + I_i(t)]} + noise

        x_i(t=0) is sampled from standard normal distribution
        W_ij can be sampled from 1) standard normal distribution 2) cluster 3) nearest neighbor
    """
    def __init__(
        self, 
        neuron_num: int, 
        dt=0.001, 
        tau=0.025,    # momentum of the system, the larger tau is the slower the system returns back to equilibrium
        spike_neuron_num=2,
        spike_input=1,
        weight_scale = 1,
        init_scale = 1,
        total_time=30000,
        data_random_seed=42,
        weight_type="random",    # "random" or "simple"
    ):
        super().__init__()
        self.neuron_num = neuron_num
        self.dt = dt
        self.tau = tau

        self.spike_neuron_num = spike_neuron_num
        self.spike_input = spike_input

        torch.manual_seed(data_random_seed)

        self.b = torch.randn(neuron_num)    # constant input for each neuron

        if weight_type == "random":
            self.W_ij = weight_scale * torch.randn(neuron_num, neuron_num) # W_ij initialization
        elif weight_type == "random_sparse":
            # self.W_ij = torch.zeros(neuron_num, neuron_num)
            self.W_ij = weight_scale * 0.5 * torch.randn(neuron_num, neuron_num)
            for i in range(0, 5):
                self.W_ij[i][8-i] = 1
                self.W_ij[8-i][i] = 1
            for i in range(1, 6):
                self.W_ij[i][10-i] = 1
                self.W_ij[10-i][i] = 1
        elif weight_type == "sparse":
            self.W_ij = torch.zeros(neuron_num, neuron_num)
            for i in range(0, 5):
                self.W_ij[i][8-i] = 1
                self.W_ij[8-i][i] = 1
            for i in range(1, 6):
                self.W_ij[i][10-i] = 1
                self.W_ij[10-i][i] = 1
        elif weight_type == "low_rank":
            rank = 18
            eigenvalues = torch.normal(0, 2, size=(rank,))
            eigenvectors_1 = torch.normal(0, 1, size=(rank, neuron_num))
            eigenvectors_2 = torch.normal(0, 1, size=(rank, neuron_num))

            self.W_ij = torch.zeros((neuron_num, neuron_num))
            for i in range(rank):
                self.W_ij += eigenvalues[i] * (eigenvectors_1[i].view(neuron_num,1)@eigenvectors_2[i].view(1,neuron_num))
        else:
            self.W_ij = tools.construct_weight_matrix(neuron_num, type=weight_type)

        self.x_t = init_scale * torch.randn(neuron_num)    # x_(t=0) initialization

        self.selected_neurons = torch.randint(low=0, high=neuron_num, size=(total_time, 2))

    def forward(self, current_time_step):
        # For each time step, randomly choose 2 neurons to add input=1
        selected = self.selected_neurons[current_time_step]
        I_t = torch.zeros(self.neuron_num)
        I_t[selected] = self.spike_input

        # x_t_1 = (1 - self.dt/self.tau) * self.x_t - self.dt/self.tau * F.tanh(self.W_ij @ self.x_t + I_t) + torch.randn(self.neuron_num) 
        # x_t_1 = (1 - self.dt/self.tau) * self.x_t - self.dt/self.tau * (self.W_ij @ F.tanh(self.x_t))
        # x_t_1 = self.W_ij @ F.tanh(self.x_t)
        x_t_1 = F.tanh((self.W_ij @ self.x_t) + self.b)
        self.x_t = x_t_1
        return x_t_1   # this is a vector of size neuron_num
    


def generate_simulation_data(
    neuron_num = 10,
    dt = 0.001,
    tau = 0.025,
    spike_neuron_num=2,
    spike_input=1,
    weight_scale = 1,
    init_scale = 1,
    total_time = 30000,
    data_random_seed=42,
    weight_type="random",
    train_data_size = 20000,
    window_size = 200,
    batch_size = 32,
    num_workers: int = 6, 
    shuffle: bool = False,
    split_ratio = 0.8,
    task_type = "reconstruction",    # "reconstruction" or "prediction" or "baseline_2"
    predict_window_size = 100,
    data_type = "wuwei", #"ziyu", "c_elegans"
) -> DataLoader:
    """
    Generate dataset.
    Return dataloaders and ground truth weight matrix.
    """

    # Simulate 10 neuron data for 30,000 time steps

    # torch.manual_seed(data_random_seed)

    if data_type == "ziyu":
        data, S = ziyu_data_simulator()
        total_time = data.shape[1]
        neuron_num = data.shape[0]
        data = torch.from_numpy(data).float()

    elif data_type == "wuwei":
        simulator = data_simulator(
            neuron_num=neuron_num, 
            dt=dt, 
            tau=tau,  
            spike_neuron_num=spike_neuron_num, 
            spike_input=spike_input,
            weight_scale=weight_scale,
            init_scale=init_scale,
            total_time=total_time,
            data_random_seed=data_random_seed,
            weight_type=weight_type
        )

        data = []

        for t in range(total_time):
            x_t = simulator.forward(t)
            x_t = x_t.view(-1, 1)
            data.append(x_t)
        data = torch.cat(data, dim=1).float()

    elif data_type == "c_elegans":
        data, S_E, S_Chem = c_elegans_data_simulator()
        total_time = data.shape[1]
        neuron_num = data.shape[0]
        data = torch.from_numpy(data).float()

        train_data_size = 1000

    print(data.shape)

    # Normalize on entire dataset
    # mean = torch.mean(data, dim=1).view(-1, 1)
    # std = torch.std(data, dim=1).view(-1, 1)
    # data = (data - mean) / std


    # Construct train/val data after simulation (time width=200)
    #
    # Train data: 
    # - first 80% time steps (30,000 * 0.8 = 24,000)
    # - randomly sample N indices as the start positions
    #
    # Val data: 
    # - last 20% time steps (30,000 * 0.2 = 6,000)
    # - N sliding windows

    train_data_length = int(total_time * split_ratio)
    train_data = data[:, :train_data_length]
    val_data = data[:, train_data_length:]

    if (task_type == "reconstruction") or (task_type == "prediction"):
        val_data_size = val_data.shape[1] - window_size + 1
        val_start_indices = torch.arange(val_data_size)

        val_data_result = []
        for i in range(val_data_size):
            index = val_start_indices[i]
            sample = val_data[:, index:index+window_size]
            val_data_result.append(sample.view(1, neuron_num, window_size))
        val_data = torch.cat(val_data_result, dim=0)

        print("val_data.shape: ", val_data.shape)
        print(val_data[-1])

        train_start_indices = torch.randint(low=0, high=train_data_length-window_size+1, size=(train_data_size,))
        train_datar_result = []
        for i in range(train_data_size):
            index = train_start_indices[i]
            sample = train_data[:, index:index+window_size]
            train_datar_result.append(sample.view(1, neuron_num, window_size))
        train_data = torch.cat(train_datar_result, dim=0)

    elif task_type == "baseline_2":  
        # Baseline_2 takes in activity from one previous time step to predict for the next time step
        train_x = train_data[:, :-1].transpose(0, 1)
        train_y = train_data[:, 1:].transpose(0, 1)

        val_x = val_data[:, :-1].transpose(0, 1)
        val_y = val_data[:, 1:].transpose(0, 1)


    if task_type == "reconstruction":
        train_dataset = TensorDataset(train_data)
        val_dataset = TensorDataset(val_data)
    elif task_type == "prediction":
        train_dataset = TensorDataset(train_data[:, :, :-predict_window_size], train_data[:, :, -predict_window_size:])
        val_dataset = TensorDataset(val_data[:, :, :-predict_window_size], val_data[:, :, -predict_window_size:])
    elif task_type == "baseline_2":
        train_dataset = TensorDataset(train_x, train_y)
        val_dataset = TensorDataset(val_x, val_y)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    if data_type == "ziyu":
        return train_dataloader, val_dataloader, torch.from_numpy(S)
    elif data_type == "wuwei":
        return train_dataloader, val_dataloader, simulator.W_ij
    elif data_type == "c_elegans":
        return train_dataloader, val_dataloader, (torch.from_numpy(S_E), torch.from_numpy(S_Chem))



def ziyu_data_simulator():
    # Set random seed for reproducibility
    np.random.seed(0)

    Ne = 4
    Ni = 1

    re = np.random.rand(Ne, 1)
    ri = np.random.rand(Ni, 1)

    a = np.concatenate((0.02 * np.ones((Ne, 1)), 0.02 + 0.08 * ri), axis=0)
    b = -0.1 * np.ones((Ne + Ni, 1))
    c = np.concatenate((-65 + 15 * re**2, -65 * np.ones((Ni, 1))), axis=0)
    d = np.concatenate((8 - 6 * re**2, 2 * np.ones((Ni, 1))), axis=0)

    params = np.concatenate((a, b, c, d), axis=1)

    S0 = 5

    # Specify connectivity
    S = np.zeros((Ne + Ni, Ne + Ni))
    S[0, 3] = 1
    S[3, 0] = 1
    S[1, 2] = 1
    S[2, 1] = 1
    S[2, 3] = 1
    S[3, 2] = 1
    S[1, 4] = -1
    S[4, 1] = 1
    S = S0 * S

    v = -65 * np.ones((Ne + Ni, 1))
    u = b * v
    firings = []
    V = [v]
    U = [u]
    inps = []

    for t in range(1, 5001):  # Simulation of 5000 ms
        I = np.concatenate((5 * np.ones((Ne, 1)), 5 * np.ones((Ni, 1))), axis=0)
        inps.append(I)
        fired = np.where(v >= 30)[0]
        firings.extend([(t + 0 * f, f) for f in fired])
        v[fired] = c[fired]
        u[fired] = u[fired] + d[fired]
        I = I + np.sum(S[:, fired], axis=1).reshape(-1, 1)
        v = v + 0.5 * (0.04 * v**2 + 4.1 * v + 108 - u + I)
        v = v + 0.5 * (0.04 * v**2 + 4.1 * v + 108 - u + I)
        u = u + a * (b * v - u)
        V.append(v)
        U.append(u)

    firings = np.array(firings)
    
    result = np.zeros((5, 5000))
    result[0][firings[firings[:,1]==0]-1] = 1
    result[1][firings[firings[:,1]==1]-1] = 1
    result[2][firings[firings[:,1]==2]-1] = 1
    result[3][firings[firings[:,1]==3]-1] = 1
    result[4][firings[firings[:,1]==4]-1] = 1

    result[0] = 5 * result[0]
    result[1] = 5 * result[1]
    result[2] = 5 * result[2]
    result[3] = 5 * result[3]
    result[4] = 2.5 * result[4]

    return result, S





def c_elegans_data_simulator():
    N_dataset = 21   # the number of worms
    N_cell = 189     # the number of neurons
    T = 960          # the number of time steps
    N_length = 109
    odor_channels = 3
    T_start = 160
    trace_datasets = np.zeros((N_dataset, N_cell, T))
    odor_datasets = np.zeros((N_dataset, odor_channels, T))
    name_list = []

    # .mat data load
    basepath = '../data/'
    mat_fname = pjoin(basepath, 'all_traces_Heads_new.mat')
    trace_variable = sio.loadmat(mat_fname)
    #trace_arr = trace_variable['norm_traces']
    trace_arr = trace_variable['traces']
    is_L = trace_variable['is_L']
    neurons_name = trace_variable['neurons']
    stim_names = trace_variable["stim_names"]
    stimulate_seconds = trace_variable['stim_times']
    stims = trace_variable['stims']
    # multiple trace datasets concatnate
    for idata in range(N_dataset):
        ineuron = 0
        for ifile in range(N_length):
            if trace_arr[ifile][0].shape[1] == 42:
                data = trace_arr[ifile][0][0][idata]
                if data.shape[0] < 1:
                    trace_datasets[idata][ineuron][:] = np.nan
                else:
                    trace_datasets[idata][ineuron][0:data[0].shape[0]] = data[0]
                ineuron+= 1
                data = trace_arr[ifile][0][0][idata + 21]
                if data.shape[0] < 1:
                    trace_datasets[idata][ineuron][:] = np.nan
                else:
                    trace_datasets[idata][ineuron][0:data[0].shape[0]] = data[0]
                ineuron+= 1
            else:
                data = trace_arr[ifile][0][0][idata]
                if data.shape[0] < 1:
                    trace_datasets[idata][ineuron][:] = np.nan
                else:
                    trace_datasets[idata][ineuron][0:data[0].shape[0]] = data[0]
                ineuron+= 1

    # neural activity target
    activity_worms = trace_datasets[:,:, T_start:] + 2      # ********************
    name_list = []
    for ifile in range(N_length):
        if is_L[ifile][0][0].shape[0] == 42:
            name_list.append(neurons_name[ifile][0][0] + 'L')
            name_list.append(neurons_name[ifile][0][0] + 'R')
        else:
            name_list.append(neurons_name[ifile][0][0])
    activity_list = name_list

    step = 0.25
    time = np.arange(start = 0, stop = T * step , step = step)
    # odor list
    odor_list = ['butanone','pentanedione','NaCL']
    # multiple odor datasets concatnate
    for idata in range(N_dataset):
        for it_stimu in range(stimulate_seconds.shape[0]):
            tim1_ind = time>stimulate_seconds[it_stimu][0]
            tim2_ind = time<stimulate_seconds[it_stimu][1]
            odor_on = np.multiply(tim1_ind.astype(np.int64),tim2_ind.astype(np.int64))
            stim_odor = stims[idata][it_stimu] - 1
            odor_datasets[idata][stim_odor][:] = odor_on
                    
    odor_worms = odor_datasets[:,:, T_start:]

    # remove the last time step in activity_worms
    activity_worms = activity_worms[:, :, :-1]

    # build a dictionary from index to neuron name
    index_to_neuron = {}
    for i in range(len(activity_list)):
        index_to_neuron[i] = activity_list[i]

    # build a dictionary from neuron name to index
    neuron_to_index = {}
    for i in range(len(activity_list)):
        neuron_to_index[activity_list[i]] = i


    connectivity_df = pd.read_csv('../data/white_neuron_connect.csv', skipinitialspace = True)

    connectivity_result_E = np.zeros((N_cell, N_cell))
    connectivity_result_Chem = np.zeros((N_cell, N_cell))
    # Loop through the rows of the connectivity df
    for i in range(connectivity_df.shape[0]):
        if connectivity_df.iloc[i]['Type'] == 'EJ':
            # get the pre and post neuron names
            pre_neuron = connectivity_df.iloc[i]['Neuron 1']
            post_neuron = connectivity_df.iloc[i]['Neuron 2']
            if pre_neuron not in neuron_to_index or post_neuron not in neuron_to_index:
                continue
            # get the pre and post neuron indices
            pre_index = neuron_to_index[pre_neuron]
            post_index = neuron_to_index[post_neuron]
            # set the connectivity matrix
            connectivity_result_E[pre_index][post_index] = connectivity_df.iloc[i]['Nbr']

        elif connectivity_df.iloc[i]['Type'] == 'S' or connectivity_df.iloc[i]['Type'] == 'SP':
            pre_neuron = connectivity_df.iloc[i]['Neuron 2']
            post_neuron = connectivity_df.iloc[i]['Neuron 1']
            if pre_neuron not in neuron_to_index or post_neuron not in neuron_to_index:
                continue
            pre_index = neuron_to_index[pre_neuron]
            post_index = neuron_to_index[post_neuron]
            connectivity_result_Chem[pre_index][post_index] = connectivity_df.iloc[i]['Nbr']

        else:
            pre_neuron = connectivity_df.iloc[i]['Neuron 1']
            post_neuron = connectivity_df.iloc[i]['Neuron 2']
            if pre_neuron not in neuron_to_index or post_neuron not in neuron_to_index:
                continue
            pre_index = neuron_to_index[pre_neuron]
            post_index = neuron_to_index[post_neuron]
            connectivity_result_Chem[pre_index][post_index] = connectivity_df.iloc[i]['Nbr']

    return activity_worms[0], connectivity_result_E, connectivity_result_Chem