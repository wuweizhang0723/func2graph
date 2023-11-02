import numpy as np
import torch
import pandas as pd
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from scipy import stats, signal



# Implement two ways to look for weight matrix after training is done
# 1) take average of all N attention outputs as weight matrix
# 2) uses sliding windows so that we can visualize if the attention output is smoothly and continuously changed


def get_avg_attention(dataloader, predict_mode_model, checkpoint_path, neuron_num=10):
    trainer = pl.Trainer(
        devices=[0],
        accelerator="gpu",
        benchmark=False,
        profiler="simple",
    )

    # attentions: batch_num * batch_size * neuron_num * neuron_num
    results = trainer.predict(predict_mode_model, dataloaders=[dataloader], ckpt_path=checkpoint_path)

    predictions = []
    ground_truths = []
    attentions = []
    for i in range(len(results)):
        x_hat = results[i][0]    # batch_size * (neuron_num*time)
        x = results[i][1]
        attention = results[i][2]
        attention = attention.view(-1, neuron_num, neuron_num)

        predictions.append(x_hat)
        ground_truths.append(x)
        attentions.append(attention)
    
    predictions = torch.cat(predictions, dim=0).cpu().numpy()  # N * neuron_num * window_size
    ground_truths = torch.cat(ground_truths, dim=0).cpu().numpy()  # N * neuron_num * window_size
    attentions = torch.cat(attentions, dim=0).cpu().numpy()    # N * neuron_num * neuron_num
    
    # # get average attention across 
    avg_attention = np.mean(attentions, axis=0)   # neuron_num * neuron_num
    return predictions, ground_truths, avg_attention, attentions




# Construct weight matrix -------------------------------
# This is used in data generation procedure to decide what type of weight matrix to use
#
# def construct_weight_matrix(neuron_num, type='nearest_neighbor'):
#     if type=='nearest_neighbor':
#         # random sample neuron_num points on 1-D line
#         # then construct weight matrix based on the distance between each pair of points
#         positions = torch.rand(neuron_num)
#         distances = torch.zeros(neuron_num, neuron_num)
#         weight_matrix = torch.zeros(neuron_num, neuron_num)
#         for i in range(neuron_num):
#             for j in range(i+1, neuron_num):
#                 distances[i, j] = torch.abs(positions[i] - positions[j])
#                 distances[j, i] = distances[i, j]
#                 weight_matrix[i, j] = 1 / distances[i, j]
#                 weight_matrix[j, i] = weight_matrix[i, j]
        
#         # normalize weight matrix
#         mean = torch.mean(weight_matrix)
#         std = torch.std(weight_matrix)
#         weight_matrix = (weight_matrix - mean) / std

#         # Contol sparsity
#         # weight_matrix[weight_matrix < 0.1] = 0
#     return weight_matrix




def group_connectivity_matrix_by_cell_type(estimated_connectivity_matrix, neuron_types):

    # Create an index of for each cell type, and also the list of cells (their original indices) in each cell type ###############
    
    cell_type2cell_type_index = {}
    cell_in_cell_type = {}
    cell_type_index2cell_type = {}
    cell_type_count = {}

    cell_type_index_count = 0
    for i in range(len(neuron_types)):
        # if cell type doesn't exist, add it to the order
        if neuron_types[i] not in cell_type2cell_type_index:
            cell_type2cell_type_index[neuron_types[i]] = cell_type_index_count
            cell_type_index_count += 1

        if neuron_types[i] not in cell_in_cell_type:
            cell_in_cell_type[neuron_types[i]] = [i]
        else:
            cell_in_cell_type[neuron_types[i]].append(i)

    for cell_type in cell_type2cell_type_index:
        cell_type_index2cell_type[cell_type2cell_type_index[cell_type]] = cell_type

    for cell_type in cell_type2cell_type_index:
        cell_type_count[cell_type] = len(cell_in_cell_type[cell_type])


    # Create a new correlation matrix based on the cell type order #############################################################

    connectivity_matrix_new = np.zeros((len(neuron_types), len(neuron_types)))

    current_cell_type_index_i = 0
    index_of_cell_in_cell_type_i = 0
    for i in range(len(neuron_types)):
        if index_of_cell_in_cell_type_i >= len(cell_in_cell_type[cell_type_index2cell_type[current_cell_type_index_i]]):
            current_cell_type_index_i += 1
            index_of_cell_in_cell_type_i = 0

        current_cell_type_i = cell_type_index2cell_type[current_cell_type_index_i]

        # get the index of the cell in the original correlation matrix
        old_i = cell_in_cell_type[current_cell_type_i][index_of_cell_in_cell_type_i]  
        index_of_cell_in_cell_type_i += 1

        current_cell_type_index_j = 0
        index_of_cell_in_cell_type_j = 0
        for j in range(len(neuron_types)):
            if index_of_cell_in_cell_type_j >= len(cell_in_cell_type[cell_type_index2cell_type[current_cell_type_index_j]]):
                current_cell_type_index_j += 1
                index_of_cell_in_cell_type_j = 0

            current_cell_type_j = cell_type_index2cell_type[current_cell_type_index_j]

            # get the index of the cell in the original correlation matrix
            old_j = cell_in_cell_type[current_cell_type_j][index_of_cell_in_cell_type_j]
            index_of_cell_in_cell_type_j += 1

            connectivity_matrix_new[i, j] = estimated_connectivity_matrix[old_i, old_j]


        # Calculate the average correlation for each cell type ##################################################################

    connectivity_matrix_cell_type_level = np.zeros((len(cell_type2cell_type_index), len(cell_type2cell_type_index)))

    accumulated_num_cells_i = 0
    for i in range(len(cell_type2cell_type_index)):
        old_i_start = accumulated_num_cells_i
        accumulated_num_cells_i += cell_type_count[cell_type_index2cell_type[i]]

        accumulated_num_cells_j = 0
        for j in range(len(cell_type2cell_type_index)):
            old_j_start = accumulated_num_cells_j
            accumulated_num_cells_j += cell_type_count[cell_type_index2cell_type[j]]

            # (Only for correlation) Remember to correct the denominator to be (total elements - # of cells in each cluster) when calculate for digonal elements
            num_elements_i = accumulated_num_cells_i - old_i_start
            num_elements_j = accumulated_num_cells_j - old_j_start
            total_num_elements = num_elements_i * num_elements_j
            # if i == j:
            #     total_num_elements = total_num_elements - cell_type_count[cell_type_index2cell_type[i]]

            connectivity_matrix_cell_type_level[i, j] = np.sum(connectivity_matrix_new[old_i_start : accumulated_num_cells_i, old_j_start : accumulated_num_cells_j]) / total_num_elements

    return connectivity_matrix_new, connectivity_matrix_cell_type_level, cell_type2cell_type_index





########################################################################################
########################################################################################
## For mouse data with multiple sessions.
########################################################################################
########################################################################################

def assign_unique_neuron_ids(all_sessions_original_UniqueID, num_neurons_per_session):
    """
    all_sessions_original_UniqueID: a concatenated list of the original UniqueID from all sessions

    Return:
    all_sessions_new_UniqueID: a list of sessions new UniqueID, each session is a 1D array of shape num_neurons
    """

    # first reassign ID starting from 0 to those non-NaN neurons
    # same IDs should be assigned to neurons that have the same original UniqueID
    non_nan_values = all_sessions_original_UniqueID[~np.isnan(all_sessions_original_UniqueID)]
    unique_non_nan_values = np.unique(non_nan_values)
    id_mapping = {unique_non_nan_values[i]: i for i in range(len(unique_non_nan_values))}

    new_ids = [id_mapping[non_nan_values[i]] for i in range(len(non_nan_values))]
    all_sessions_new_UniqueID = np.copy(all_sessions_original_UniqueID)
    all_sessions_new_UniqueID[~np.isnan(all_sessions_new_UniqueID)] = new_ids

    # then assign new IDs to those NaN neurons
    num_unique_non_nan = unique_non_nan_values.shape[0]     # new IDs start from num_unqiue_non_nan
    num_nan = np.sum(np.isnan(all_sessions_original_UniqueID))           # new IDs end with num_non_nan + num_nan -1

    new_ids = np.arange(num_unique_non_nan, num_unique_non_nan + num_nan)
    all_sessions_new_UniqueID[np.isnan(all_sessions_new_UniqueID)] = new_ids

    # Segment all_sessions_new_UniqueID into sessions
    all_sessions_new_UniqueID = np.split(all_sessions_new_UniqueID, np.cumsum(num_neurons_per_session)[:-1])

    num_unique_neurons = num_unique_non_nan + num_nan

    return all_sessions_new_UniqueID, num_unique_neurons    # shape: num_sessions x num_neurons_per_session



def assign_unique_cell_type_ids(all_sessions_original_cell_type, num_neurons_per_session):
    """
    all_sessions_original_cell_type: a concatenated list of the original cell types from all sessions (raw cell types)

    Return:
    all_sessions_new_cell_type: a list of sessions new cell type, each session is a 1D array of shape num_neurons
    """
    # Get the first level of cell types
    neuron_types_result = []
    for i in range(len(all_sessions_original_cell_type)):
        # split by "-"
        neuron_types_result.append(all_sessions_original_cell_type[i].split("-")[0])
    all_sessions_original_cell_type = neuron_types_result

    unique_cell_types = list(set(all_sessions_original_cell_type))
    # Assign IDs to cell types
    cell_type2id = {unique_cell_types[i]: i for i in range(len(unique_cell_types))}

    # Get new cell type IDs
    all_sessions_new_cell_type_id = np.zeros(len(all_sessions_original_cell_type))
    for i in range(len(all_sessions_original_cell_type)):
        all_sessions_new_cell_type_id[i] = cell_type2id[all_sessions_original_cell_type[i]]

    # Segment all_sessions_new_cell_type_id into sessions
    all_sessions_new_cell_type_id = np.split(all_sessions_new_cell_type_id, np.cumsum(num_neurons_per_session)[:-1])

    return all_sessions_new_cell_type_id, cell_type2id     # shape: num_sessions x num_neurons_per_session



def sliding_windows(all_sessions_acitvity, all_sessions_new_UniqueID, all_sessions_new_cell_type_id, window_size):
    """
    (can be from TRAIN or VAL set)
    all_sessions_acitvity: a list of sessions activity, each session is a 2D array of shape num_neurons x num_frames
    all_sessions_new_UniqueID: a list of sessions new UniqueID, each session is a 1D array of shape num_neurons
    all_sessions_new_cell_type_id: a list of sessions new cell type id, each session is a 1D array of shape num_neurons

    Return:
    - all_sessions_activity_windows:
        a list of sessions activity windows, each session is a 3D array of shape num_windows x num_neurons x window_size
    - all_sessions_new_UniqueID_windows:
        a list of sessions new UniqueID windows, each session is a 2D array of shape num_windows x num_neurons (each row should be the same)
    - all_sessions_new_cell_type_id_windows:
        a list of sessions new cell type id windows, each session is a 2D array of shape num_windows x num_neurons (each row should be the same)
    """

    all_sessions_activity_windows = []
    all_sessions_new_UniqueID_windows = []
    all_sessions_new_cell_type_id_windows = []

    for i in range(len(all_sessions_acitvity)):
        num_neurons = all_sessions_acitvity[i].shape[0]
        num_frames = all_sessions_acitvity[i].shape[1]
        num_windows = num_frames - window_size + 1

        # activity
        activity_windows = np.zeros((num_windows, num_neurons, window_size))
        for j in range(num_windows):
            activity_windows[j] = all_sessions_acitvity[i][:, j:j+window_size]
        all_sessions_activity_windows.append(activity_windows)

        # UniqueID
        UniqueID_windows = np.zeros((num_windows, num_neurons))
        for j in range(num_windows):
            UniqueID_windows[j] = all_sessions_new_UniqueID[i]
        all_sessions_new_UniqueID_windows.append(UniqueID_windows)

        # cell type id
        cell_type_id_windows = np.zeros((num_windows, num_neurons))
        for j in range(num_windows):
            cell_type_id_windows[j] = all_sessions_new_cell_type_id[i]
        all_sessions_new_cell_type_id_windows.append(cell_type_id_windows)

    return all_sessions_activity_windows, all_sessions_new_UniqueID_windows, all_sessions_new_cell_type_id_windows