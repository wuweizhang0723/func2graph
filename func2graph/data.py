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
        tau=1,    # 1, 2, 3, 4, 5 ...
        weight_scale = 0.2,
        init_scale = 0.2,
        error_scale = 1,
        total_time=30000,
        data_random_seed=42,
        weight_type="cell_type",    # "random" or "simple" or "cell_type"
        test=False,
    ):
        super().__init__()
        self.test = test
        self.error_scale = error_scale

        self.neuron_num = neuron_num
        self.tau = tau

        torch.manual_seed(data_random_seed)
        np.random.seed(data_random_seed)

        self.b = weight_scale * torch.randn(neuron_num)    # constant input for each neuron

        self.activity = []     # store the activity of each neuron at each time step so far（total_time x neuron_num)
        for i in range(int(tau)):
            self.activity.append(init_scale * torch.randn(neuron_num))

        if weight_type == "random":
            self.W_ij = weight_scale * torch.randn(neuron_num, neuron_num) # W_ij initialization
        elif weight_type == "sparse":
            self.W_ij = torch.zeros(neuron_num, neuron_num)
            for i in range(0, 5):
                self.W_ij[i][8-i] = 1
                self.W_ij[8-i][i] = 1
            for i in range(1, 6):
                self.W_ij[i][10-i] = 1
                self.W_ij[10-i][i] = 1
        elif weight_type == "low_rank":
            rank = 18  ####################
            eigenvalues = torch.normal(0, 2, size=(rank,))
            eigenvectors_1 = torch.normal(0, 1, size=(rank, neuron_num))
            eigenvectors_2 = torch.normal(0, 1, size=(rank, neuron_num))

            self.W_ij = torch.zeros((neuron_num, neuron_num))
            for i in range(rank):
                self.W_ij += eigenvalues[i] * (eigenvectors_1[i].view(neuron_num,1)@eigenvectors_2[i].view(1,neuron_num))
        elif weight_type == "cell_type":
            self.W_ij, self.cell_type_order, self.cell_type_ids, self.cell_type_count = tools.construct_weight_matrix_cell_type(neuron_num)
            self.W_ij = weight_scale * self.W_ij

    def forward(self, current_time_step):
        signal = F.tanh((self.W_ij @ self.activity[-int(self.tau)]) + self.b)
        e = torch.normal(0,1,size=(self.neuron_num,)) * self.error_scale
        x_t_1 = signal + e

        self.activity.append(x_t_1)

        if self.test == True:
            return signal, e
        return x_t_1   # this is a vector of size neuron_num
    


def generate_simulation_data(
    neuron_num = 10,
    tau = 0.025,
    weight_scale = 1,
    init_scale = 1,
    total_time = 30000,
    data_random_seed=42,
    weight_type="cell_type",
    window_size = 200, # -------------------------------
    batch_size = 32,
    num_workers: int = 6, 
    shuffle: bool = False,
    split_ratio = 0.8,
    task_type = "prediction",    # "reconstruction" or "prediction" or "GLM_sim_exp" or "GLM_sim_tanh"
    predict_window_size = 100,
    data_type = "wuwei", #"ziyu", "c_elegans", "mouse"
    spatial_partial_measurement = 200,  # the number of neurons to measured, between 0 and neuron_num
) -> DataLoader:
    """
    Generate dataset.
    Return dataloaders and ground truth weight matrix.
    """

    # Simulate 10 neuron data for 30,000 time steps

    # torch.manual_seed(data_random_seed)

    if data_type == "ziyu":
        v_normed_alltimes = pd.read_csv('../data/Ziyu/200/v_normed_alltimes.txt', header=None, sep=',')
        v_normed_alltimes = v_normed_alltimes.to_numpy()

        total_time = v_normed_alltimes.shape[1]
        neuron_num = v_normed_alltimes.shape[0]
        data = torch.from_numpy(v_normed_alltimes).float()

        W = pd.read_csv('../data/Ziyu/200/connectivity.txt', header=None, sep=',')
        W = W.to_numpy()

    elif data_type == "wuwei":
        simulator = data_simulator(
            neuron_num=neuron_num, 
            tau=tau,  
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

        train_data_size = 1000  ########################

    if spatial_partial_measurement != neuron_num:
        # choose a subset of neurons to measure, without replacement
        idx = torch.randperm(neuron_num)[:spatial_partial_measurement]
        # sort the indices
        idx = torch.sort(idx)[0] # !!!!!!
        print('idx: ', idx)
        data = data[idx, :]
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
            if spatial_partial_measurement != neuron_num:
                val_data_result.append(sample.view(1, spatial_partial_measurement, window_size))
            else:
                val_data_result.append(sample.view(1, neuron_num, window_size))
        val_data = torch.cat(val_data_result, dim=0)

        print("val_data.shape: ", val_data.shape)
        print(val_data[-1])

        # train_start_indices = torch.randint(low=0, high=train_data_length-window_size+1, size=(train_data_size,))
        # train_datar_result = []
        # for i in range(train_data_size):
        #     index = train_start_indices[i]
        #     sample = train_data[:, index:index+window_size]
        #     train_datar_result.append(sample.view(1, neuron_num, window_size))
        # train_data = torch.cat(train_datar_result, dim=0)

        train_data_size = train_data_length - window_size + 1
        train_start_indices = torch.arange(train_data_size)

        train_data_result = []
        for i in range(train_data_size):
            index = train_start_indices[i]
            sample = train_data[:, index:index+window_size]
            if spatial_partial_measurement != neuron_num:
                train_data_result.append(sample.view(1, spatial_partial_measurement, window_size))
            else:
                train_data_result.append(sample.view(1, neuron_num, window_size))
        train_data = torch.cat(train_data_result, dim=0)



    elif (task_type == "GLM_sim_exp") or (task_type == "GLM_sim_tanh"):  
        # Baseline_2 takes in activity from one previous time step to predict for the next time step
        train_x = train_data[:, :-1].transpose(0, 1)
        train_y = train_data[:, 1:].transpose(0, 1)

        val_x = val_data[:, :-1].transpose(0, 1)
        val_y = val_data[:, 1:].transpose(0, 1)

        if task_type == "GLM_sim_exp":    
            # find the min value and if it is negative, add abs(min val)
            min_val = torch.min(data)
            print('min_val: ', min_val)

            print(torch.min(train_x), torch.min(train_y), torch.min(val_x), torch.min(val_y))

            print('abs min_val: ', torch.abs(min_val))

            if min_val < 0:
                train_x = torch.abs(min_val) + train_x
                train_y = torch.abs(min_val) + train_y
                val_x = torch.abs(min_val) + val_x
                val_y = torch.abs(min_val) + val_y

                # destd
                # std = torch.std(data)
                # train_x /= std
                # train_y /= std
                # val_x /= std
                # val_y /= std
            
            print('min_val_now: ', torch.min(train_x), torch.min(train_y), torch.min(val_x), torch.min(val_y))
            print('max_val_now: ', torch.max(train_x), torch.max(train_y), torch.max(val_x), torch.max(val_y))
            


    if task_type == "reconstruction":
        train_dataset = TensorDataset(train_data)
        val_dataset = TensorDataset(val_data)
    elif task_type == "prediction":
        train_dataset = TensorDataset(train_data[:, :, :-predict_window_size], train_data[:, :, -predict_window_size:])
        val_dataset = TensorDataset(val_data[:, :, :-predict_window_size], val_data[:, :, -predict_window_size:])
    elif (task_type == "GLM_sim_exp") or (task_type == "GLM_sim_tanh"):
        train_dataset = TensorDataset(train_x, train_y)
        val_dataset = TensorDataset(val_x, val_y)


    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    if spatial_partial_measurement == neuron_num:
        if data_type == "ziyu":
            return train_dataloader, val_dataloader, torch.from_numpy(v_normed_alltimes[:,0]), torch.from_numpy(W)
        elif data_type == "wuwei":
            return train_dataloader, val_dataloader, simulator.W_ij, simulator.b, simulator.cell_type_ids, simulator.cell_type_order, simulator.cell_type_count
        elif data_type == "c_elegans":
            return train_dataloader, val_dataloader, (torch.from_numpy(S_E), torch.from_numpy(S_Chem))
    else:
        if data_type == "wuwei":
            new_W_ij = torch.zeros((spatial_partial_measurement, spatial_partial_measurement))
            for i in range(spatial_partial_measurement):
                for j in range(spatial_partial_measurement):
                    new_W_ij[i][j] = simulator.W_ij[idx[i]][idx[j]]

            new_cell_type_ids = []
            for i in range(spatial_partial_measurement):
                new_cell_type_ids.append(simulator.cell_type_ids[idx[i]])

            id2cell_type = {0:'EC', 1:'Pv', 2:'Sst', 3:'Vip'}
            new_cell_type_count = {'EC':0, 'Pv':0, 'Sst':0, 'Vip':0}
            for i in range(spatial_partial_measurement):
                cell_type = id2cell_type[new_cell_type_ids[i]]
                new_cell_type_count[cell_type] += 1

            return train_dataloader, val_dataloader, new_W_ij, new_cell_type_ids, simulator.cell_type_order, new_cell_type_count
    







# C_elegans Data ------------------------------------------------------------------------


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



# Mouse Data --------------------------------------------------------------------------


def load_mouse_data_session(directory, date_exp, input_setting, normalization):
    gene_count = np.load(directory + date_exp + 'neuron.gene_count.npy')
    UniqueID = np.load(directory + date_exp + 'neuron.UniqueID.npy')

    with open(directory + date_exp + 'neuron.ttype.txt') as f:
        neuron_ttypes_raw  = f.readlines()
    neuron_ttypes = []
    for neuron_ttype in neuron_ttypes_raw:
        neuron_ttypes.append(neuron_ttype.strip())

    frame_states = np.load(directory + date_exp + input_setting + 'frame.states.npy')
    frame_times = np.load(directory + date_exp + input_setting + 'frame.times.npy')
    frame_activity = np.load(directory + date_exp + input_setting + 'frame.neuralActivity.npy')

    # eye_size = np.load(directory + date_exp + input_setting + 'eye.size.npy')
    # eye_times = np.load(directory + date_exp + input_setting + 'eye.times.npy')

    # if np.isnan(np.mean(eye_size)):
    #     eye_size = interpolate_nans(eye_size[:,0])[:,np.newaxis]

    # times = frame_times.shape[0]
    # eye_times_list = list(eye_times[:,0])
    # eye_size_resize = np.zeros((times, 1))
    # for t in range(times):
    #     index = bisect(eye_times_list, frame_times[t,0])
    #     if t == times - 1:
    #         eye_size_resize[t] = eye_size[index - 1, 0]
    #     else:
    #         eye_size_resize[t] = (eye_size[index - 1, 0] + eye_size[index, 0])/2

    # running_speed = np.load(directory + date_exp + input_setting + 'running.speed.npy')
    # running_times = np.load(directory + date_exp + input_setting + 'running.times.npy')

    # if np.isnan(np.mean(running_speed)):
    #     running_speed = interpolate_nans(running_speed[:,0])[:,np.newaxis]

    # running_times_list = list(running_times[:,0])
    # running_speed_resize = np.zeros((times, 1))
    # for t in range(times):
    #     index = bisect(running_times_list, frame_times[t,0])
    #     if t == times - 1:
    #         running_speed_resize[t] = running_speed[index - 1, 0]
    #     else:
    #         running_speed_resize[t] = (running_speed[index - 1, 0] + running_speed[index, 0])/2


    # normalization
    if normalization == 'session':
        activity_mean = np.mean(frame_activity)
        activity_std = np.std(frame_activity)

        activity_norm = (frame_activity - activity_mean)/activity_std
    elif normalization == 'destd_session':
        activity_std = np.std(frame_activity)

        activity_norm = (frame_activity) / activity_std
    elif normalization == 'neuron':
        activity_mean = np.mean(frame_activity, axis = 0)
        activity_std = np.std(frame_activity, axis = 0)

        activity_norm = (frame_activity - activity_mean)/activity_std
    elif normalization == 'log':
        activity_norm = np.log10(frame_activity + 1)
    else:
        print('no normalization')

        activity_norm = frame_activity

    # if behavior_normalization:
    #     running_speed_resize = (running_speed_resize - np.mean(running_speed_resize, axis = 0))/np.std(running_speed_resize, axis = 0)
    #     eye_size_resize = (eye_size_resize - np.mean(eye_size_resize, axis = 0))/np.std(eye_size_resize, axis = 0)

    # activity population
    # activity_population = {
    #     'running_speed': running_speed_resize,
    #     'eye_size': eye_size_resize,
    #     'frame_states': frame_states,
    #     'mean_activity': np.mean(activity_norm, axis = 1, keepdims = True),
    #     'std_activity': np.std(activity_norm, axis = 1, keepdims = True),
    #     }

    # x, y coordinate position of neurons
    # neuron_pos = np.load(directory + date_exp +'neuron.stackPosCorrected.npy')
    # print('neuron_pos.shape:', neuron_pos.shape)
    # plt.figure(figsize=(4,4))
    # plt.plot(neuron_pos[:,0],neuron_pos[:,1],'o')
    # plt.xlabel('x axis')
    # plt.ylabel('y axis')

    return np.transpose(activity_norm), frame_times, UniqueID.reshape(-1), neuron_ttypes








########################################################################################
########################################################################################
# For mouse data with multiple sessions,
# - data: batch_size x neuron_num x window_size
# - Each unique neuron is assigned a unique ID across all sessions (batch_size x neuron_num)
# - Each neuron has cell type id (batch_size x neuron_num)
# - cell_type2id is a dictionary from cell type to cell type id
########################################################################################
########################################################################################

class Mouse_All_Sessions_Dataset(TensorDataset):
    def __init__(
        self, 
        all_sessions_activity_windows,  # list of 3d tensors, each tensor is a session (num_window x n x window_size)) 
        all_sessions_new_UniqueID_windows,  # list of 2d tensors, each tensor is a session (num_window x n)
        all_sessions_new_cell_type_id_windows, # list of 2d tensors, each tensor is a session (num_window x n)
        batch_size=3,                      # real batch size !!!!!!!!!!!!!!!!!
    ):
        self.num_batch_per_session = [session.shape[0] // batch_size for session in all_sessions_activity_windows]

        self.all_batch = []
        self.all_batch_neuron_ids = []
        self.all_batch_cell_type_ids = []
        for i in range(len(self.num_batch_per_session)):      # for each session
            for j in range(self.num_batch_per_session[i]):      # for each batch
                self.all_batch.append(torch.Tensor(all_sessions_activity_windows[i][j*batch_size:(j+1)*batch_size]).float())
                self.all_batch_neuron_ids.append(torch.Tensor(all_sessions_new_UniqueID_windows[i][j*batch_size:(j+1)*batch_size]).int())
                self.all_batch_cell_type_ids.append(torch.Tensor(all_sessions_new_cell_type_id_windows[i][j*batch_size:(j+1)*batch_size]).int())

    def __getitem__(self, index):
        return self.all_batch[index], self.all_batch_neuron_ids[index], self.all_batch_cell_type_ids[index]

    def __len__(self):
        return len(self.all_batch)
    


def generate_mouse_all_sessions_data(
    input_mouse: list,    # e.g. [SB025, SB026]
    input_sessions: list,    # e.g. [[2019-10-07, 2019-10-04], [2019-10-11, 2019-10-14, 2019-10-16]]
    window_size = 200, 
    batch_size = 32,
    num_workers: int = 6, 
    shuffle: bool = False,
    normalization = 'session',
    split_ratio = 0.8,
):
    
    directory = '../data/Mouse/Bugeon/'
    input_sessions_file_path = []
    for i in range(len(input_mouse)):
        for j in range(len(input_sessions[i])):
            input_sessions_file_path.append({'date_exp': input_mouse[i] + '/' + input_sessions[i][j] + '/', 'input_setting': 'Blank/01/'})
            print(input_mouse[i] + '/' + input_sessions[i][j])

    all_sessions_original_UniqueID = []
    all_sessions_original_cell_type = []
    all_sessions_acitvity_TRAIN = []   # first 80% of the time
    all_sessions_acitvity_VAL = []
    num_neurons_per_session = []

    sessions_2_original_cell_type = []

    all_sessions_activity_flatten = []

    for i in range(len(input_sessions_file_path)):
        date_exp = input_sessions_file_path[i]['date_exp']
        input_setting = input_sessions_file_path[i]['input_setting']

        activity, frame_times, UniqueID, neuron_ttypes = load_mouse_data_session(
            directory, date_exp, input_setting, normalization="no"
        )

        all_sessions_original_UniqueID.append(UniqueID)
        all_sessions_acitvity_TRAIN.append(activity[:, :int(activity.shape[1]*split_ratio)])
        all_sessions_acitvity_VAL.append(activity[:, int(activity.shape[1]*split_ratio):])
        num_neurons_per_session.append(activity.shape[0])

        all_sessions_activity_flatten.append(activity.flatten())

        # Get the first level of cell types
        neuron_types_result = []
        for j in range(len(neuron_ttypes)):
            # split by "-"
            neuron_types_result.append(neuron_ttypes[j].split("-")[0])

        sessions_2_original_cell_type.append(neuron_types_result)
        all_sessions_original_cell_type.append(neuron_types_result)

    all_sessions_original_UniqueID = np.concatenate(all_sessions_original_UniqueID)
    all_sessions_original_cell_type = np.concatenate(all_sessions_original_cell_type)

    all_sessions_activity_flatten = np.concatenate(all_sessions_activity_flatten)
    mu = np.mean(all_sessions_activity_flatten)
    std = np.std(all_sessions_activity_flatten)

    # all_sessions normalization
    all_sessions_acitvity_TRAIN = [(session - mu) / std for session in all_sessions_acitvity_TRAIN]
    all_sessions_acitvity_VAL = [(session - mu) / std for session in all_sessions_acitvity_VAL]
    

    ##############################################
    # Construct new UniqueID and cell type id
    ##############################################
    all_sessions_new_UniqueID, num_unqiue_neurons = tools.assign_unique_neuron_ids(all_sessions_original_UniqueID, num_neurons_per_session)
    all_sessions_new_cell_type_id, cell_type_order = tools.assign_unique_cell_type_ids(all_sessions_original_cell_type, num_neurons_per_session)

    neuron_id_2_cell_type_id = np.zeros((num_unqiue_neurons,)).astype(int)
    for i in range(len(all_sessions_new_UniqueID)):
        neuron_id_2_cell_type_id[all_sessions_new_UniqueID[i].astype(int)] = all_sessions_new_cell_type_id[i]


    ##############################################
    # Construct windows
    ##############################################

    # For TRAIN
    all_sessions_activity_windows_TRAIN, all_sessions_new_UniqueID_windows_TRAIN, all_sessions_new_cell_type_id_window_TRAIN = tools.sliding_windows(
        all_sessions_acitvity_TRAIN, all_sessions_new_UniqueID, all_sessions_new_cell_type_id, window_size=window_size
    )
    # For VAL
    all_sessions_activity_windows_VAL, all_sessions_new_UniqueID_windows_VAL, all_sessions_new_cell_type_id_window_VAL = tools.sliding_windows(
        all_sessions_acitvity_VAL, all_sessions_new_UniqueID, all_sessions_new_cell_type_id, window_size=window_size
    )

    ##############################################
    # Construct dataloaders
    ##############################################
    train_dataset = Mouse_All_Sessions_Dataset(
        all_sessions_activity_windows_TRAIN, 
        all_sessions_new_UniqueID_windows_TRAIN, 
        all_sessions_new_cell_type_id_window_TRAIN, 
        batch_size=batch_size,        ###### real batch_size!!!!!!!!!!!!!!!!!!!!
    )
    val_dataset = Mouse_All_Sessions_Dataset(
        all_sessions_activity_windows_VAL, 
        all_sessions_new_UniqueID_windows_VAL, 
        all_sessions_new_cell_type_id_window_VAL, 
        batch_size=batch_size,        ###### real batch_size!!!!!!!!!!!!!!!!!!!!
    )

    num_batch_per_session_TRAIN = train_dataset.num_batch_per_session
    num_batch_per_session_VAL = val_dataset.num_batch_per_session

    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=num_workers)    # 1 is not real batch_size
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=num_workers)        # 1 is not real batch_size

    return train_dataloader, val_dataloader, num_unqiue_neurons, cell_type_order, all_sessions_new_cell_type_id, num_batch_per_session_TRAIN, num_batch_per_session_VAL, sessions_2_original_cell_type, neuron_id_2_cell_type_id






########################################################################################
########################################################################################
# Below is the data generation for GLM.
# It should return a list of dataloaders for each session.
#
# GLM is trained on ONE session at a time. x_{t+1} = \sum A_k x_{t-k}
# The input is previous K time steps and the output is the current ONE time step.
#
# - input data: batch_size x neuron_num x 1
# - output data: batch_size x neuron_num x 1
# - Each unique neuron is assigned a unique ID across ALL sessions (batch_size x neuron_num)
# - Each neuron has cell type id (batch_size x neuron_num)
# - cell_type2id is a dictionary from cell type to cell type id
########################################################################################
########################################################################################

def generate_mouse_all_sessions_data_for_GLM(
    input_mouse: list,    # e.g. [SB025, SB026]
    input_sessions: list,    # e.g. [[2019-10-07, 2019-10-04], [2019-10-11, 2019-10-14, 2019-10-16]]
    k,     # the number of tau(s) to consider, each tau corresponds to one A in GLM
    batch_size = 32,
    num_workers: int = 6, 
    shuffle: bool = False,
    normalization = 'all',    # destd, all
    split_ratio = 0.8,
):
    
    directory = '../data/Mouse/Bugeon/'
    input_sessions_file_path = []
    for i in range(len(input_mouse)):
        for j in range(len(input_sessions[i])):
            input_sessions_file_path.append({'date_exp': input_mouse[i] + '/' + input_sessions[i][j] + '/', 'input_setting': 'Blank/01/'})

    all_sessions_original_UniqueID = []
    all_sessions_original_cell_type = []
    all_sessions_acitvity_TRAIN = []   # first 80% of the time
    all_sessions_acitvity_VAL = []
    num_neurons_per_session = []

    sessions_2_original_cell_type = []

    all_sessions_activity_flatten = []

    for i in range(len(input_sessions_file_path)):
        date_exp = input_sessions_file_path[i]['date_exp']
        input_setting = input_sessions_file_path[i]['input_setting']

        activity, frame_times, UniqueID, neuron_ttypes = load_mouse_data_session(
            directory, date_exp, input_setting, normalization='no'    ###################################
        )

        all_sessions_original_UniqueID.append(UniqueID)
        all_sessions_acitvity_TRAIN.append(activity[:, :int(activity.shape[1]*split_ratio)])
        all_sessions_acitvity_VAL.append(activity[:, int(activity.shape[1]*split_ratio):])
        num_neurons_per_session.append(activity.shape[0])

        all_sessions_activity_flatten.append(activity.flatten())

        # Get the first level of cell types
        neuron_types_result = []
        for j in range(len(neuron_ttypes)):
            # split by "-"
            neuron_types_result.append(neuron_ttypes[j].split("-")[0])

        sessions_2_original_cell_type.append(neuron_types_result)
        all_sessions_original_cell_type.append(neuron_types_result)

    all_sessions_original_UniqueID = np.concatenate(all_sessions_original_UniqueID)     # flatten to 1d
    all_sessions_original_cell_type = np.concatenate(all_sessions_original_cell_type)   # flatten to 1d

    all_sessions_activity_flatten = np.concatenate(all_sessions_activity_flatten)
    mu = np.mean(all_sessions_activity_flatten)
    std = np.std(all_sessions_activity_flatten)

    if normalization == 'all':
        all_sessions_acitvity_TRAIN = [(session - mu) / std for session in all_sessions_acitvity_TRAIN]
        all_sessions_acitvity_VAL = [(session - mu) / std for session in all_sessions_acitvity_VAL]
    elif normalization == 'destd':
        all_sessions_acitvity_TRAIN = [(session) / std for session in all_sessions_acitvity_TRAIN]
        all_sessions_acitvity_VAL = [(session) / std for session in all_sessions_acitvity_VAL]


    ##############################################
    # Construct new UniqueID and cell type id
    ##############################################
    # all _essions_new_UniqueID: num_sessions x num_neurons_per_session
    # num_unqiue_neurons: a number, total number of unique neurons
    all_sessions_new_UniqueID, num_unqiue_neurons = tools.assign_unique_neuron_ids(all_sessions_original_UniqueID, num_neurons_per_session)
    # all_sessions_new_cell_type_id: num_sessions x num_neurons_per_session
    # cell_type_order: a list of cell types, the index corresponds to cell type id
    all_sessions_new_cell_type_id, cell_type_order = tools.assign_unique_cell_type_ids(all_sessions_original_cell_type, num_neurons_per_session)

    neuron_id_2_cell_type_id = np.zeros((num_unqiue_neurons,)).astype(int)
    for i in range(len(all_sessions_new_UniqueID)):
        neuron_id_2_cell_type_id[all_sessions_new_UniqueID[i].astype(int)] = all_sessions_new_cell_type_id[i]


    ##############################################
    # Construct windows
    ##############################################

    # For TRAIN
    # all_sessions_activity_windows: a list of sessions activity windows, each session is a 3D array of shape num_windows x num_neurons x window_size
    all_sessions_activity_windows_TRAIN, all_sessions_new_UniqueID_windows_TRAIN, all_sessions_new_cell_type_id_window_TRAIN = tools.sliding_windows(
        all_sessions_acitvity_TRAIN, all_sessions_new_UniqueID, all_sessions_new_cell_type_id, window_size=(k+1)   ########## TODO: check window size!!!!!!!!!!!!!
    )
    # For VAL
    all_sessions_activity_windows_VAL, all_sessions_new_UniqueID_windows_VAL, all_sessions_new_cell_type_id_window_VAL = tools.sliding_windows(
        all_sessions_acitvity_VAL, all_sessions_new_UniqueID, all_sessions_new_cell_type_id, window_size=(k+1)
    )

    ##############################################
    # Construct dataloaders
    ##############################################

    train_dataloader_list = []
    val_dataloader_list = []

    for i in range(len(input_sessions_file_path)):
        train_data = torch.Tensor(all_sessions_activity_windows_TRAIN[i]).float()
        val_data = torch.Tensor(all_sessions_activity_windows_VAL[i]).float()

        train_dataset = TensorDataset(train_data)
        val_dataset = TensorDataset(val_data)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        train_dataloader_list.append(train_dataloader)
        val_dataloader_list.append(val_dataloader)

    return train_dataloader_list, val_dataloader_list, num_unqiue_neurons, cell_type_order, all_sessions_new_cell_type_id, sessions_2_original_cell_type, neuron_id_2_cell_type_id