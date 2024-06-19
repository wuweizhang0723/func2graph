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
# This is used in data generation procedure to construct weight matrix that mimics the real mouse data with cell type information
#
def construct_weight_matrix_cell_type(neuron_num):
    cell_type_order = ['EC', 'Pvalb', 'Sst', 'Vip']
    # Let the first 76% neurons be EC, 8% neurons be Pv, 8% neurons be Sst, 8% neurons be Vip
    cell_type_ids = np.zeros(neuron_num, dtype=int)
    cell_type_ids[:int(neuron_num*0.76)] = 0
    cell_type_ids[int(neuron_num*0.76):int(neuron_num*0.84)] = 1
    cell_type_ids[int(neuron_num*0.84):int(neuron_num*0.92)] = 2
    cell_type_ids[int(neuron_num*0.92):] = 3
    
    cell_type_count = {'EC':int(neuron_num*0.76), 'Pvalb':int(neuron_num*0.08), 'Sst':int(neuron_num*0.08), 'Vip':int(neuron_num*0.08)}
    
    # construct cutoff matrix from science paper
    cutoff_matrix = np.zeros((4, 4))
    cutoff_matrix[0, 0] = 13/229
    cutoff_matrix[1, 0] = 22/53
    cutoff_matrix[2, 0] = 20/67
    cutoff_matrix[3, 0] = 11/68

    cutoff_matrix[0, 1] = 18/52
    cutoff_matrix[1, 1] = 45/114
    cutoff_matrix[2, 1] = 8/88
    cutoff_matrix[3, 1] = 0/54

    cutoff_matrix[0, 2] = 13/56
    cutoff_matrix[1, 2] = 15/84
    cutoff_matrix[2, 2] = 8/154
    cutoff_matrix[3, 2] = 25/84

    cutoff_matrix[0, 3] = 3/62
    cutoff_matrix[1, 3] = 1/54
    cutoff_matrix[2, 3] = 12/87
    cutoff_matrix[3, 3] = 2/209

    strength_matrix = np.zeros((4, 4))
    strength_matrix[0, 0] = 0.11
    strength_matrix[1, 0] = 0.27
    strength_matrix[2, 0] = 0.1
    strength_matrix[3, 0] = 0.45

    strength_matrix[0, 1] = -0.44
    strength_matrix[1, 1] = -0.47
    strength_matrix[2, 1] = -0.44
    strength_matrix[3, 1] = -0.23

    strength_matrix[0, 2] = -0.16
    strength_matrix[1, 2] = -0.18
    strength_matrix[2, 2] = -0.19
    strength_matrix[3, 2] = -0.17

    strength_matrix[0, 3] = -0.06
    strength_matrix[1, 3] = -0.10
    strength_matrix[2, 3] = -0.17
    strength_matrix[3, 3] = -0.10

    # uniformly initialize weight matrix with uniform distribution from 0 to 1
    weight_matrix = torch.rand(neuron_num, neuron_num)

    # Set elements below cutoff to 0
    for i in range(neuron_num):
        for j in range(neuron_num):
            cell_type_i = cell_type_ids[i]
            cell_type_j = cell_type_ids[j]
            if weight_matrix[i, j] > cutoff_matrix[cell_type_i, cell_type_j]:
                weight_matrix[i, j] = 0
            else:
                mean = strength_matrix[cell_type_i, cell_type_j]
                if cell_type_j == 0:
                    std = 0.1
                else:
                    std = 0.1
                weight_matrix[i, j] = torch.normal(mean, std, size=(1,))

    # weight_matrix_strength = torch.normal(0, 1, size=(neuron_num, neuron_num))
    # weight_matrix = weight_matrix * weight_matrix_strength

    # for j in range(neuron_num):
    #     cell_type_j = cell_type_ids[j]
    #     if cell_type_j == 0:
    #         weight_matrix[:, j] = torch.abs(weight_matrix[:, j])
    #     else:
    #         weight_matrix[:, j] = -torch.abs(weight_matrix[:, j])


    return weight_matrix, cell_type_order, cell_type_ids, cell_type_count



########################################################################################
########################################################################################


# This function is used to get k*k attention matrix from a N*N attention matrix
# It is for the first Attention model, which doesn't have k*k prior constraint.
#
# estimated_connectivity_matrix - N*N connectivity matrix
# neuron_id_2_cell_type_id - a list of cell type id for each neuron, where the list index is the neuron id
# 
def group_connectivity_matrix_by_cell_type(estimated_connectivity_matrix, neuron_id_2_cell_type_id):
    assert estimated_connectivity_matrix.shape[0] == estimated_connectivity_matrix.shape[1]
    assert estimated_connectivity_matrix.shape[0] == len(neuron_id_2_cell_type_id)

    # cell_type_id2cell_type is a dictionary, where the key is the cell type id, and the value is a list of neuron ids
    cell_type_id2neuron_id = {}
    for i in range(len(neuron_id_2_cell_type_id)):
        if neuron_id_2_cell_type_id[i] not in cell_type_id2neuron_id:
            cell_type_id2neuron_id[neuron_id_2_cell_type_id[i]] = [i]
        else:
            cell_type_id2neuron_id[neuron_id_2_cell_type_id[i]].append(i)


    # Create a new correlation matrix based on the cell type order #############################################################

    connectivity_matrix_new = np.zeros((len(neuron_id_2_cell_type_id), len(neuron_id_2_cell_type_id)))

    current_cell_type_index_i = 0
    index_of_cell_in_cell_type_i = 0
    for i in range(len(neuron_id_2_cell_type_id)):
        if index_of_cell_in_cell_type_i >= len(cell_type_id2neuron_id[current_cell_type_index_i]):
            current_cell_type_index_i += 1
            index_of_cell_in_cell_type_i = 0

        # get the index of the cell in the original correlation matrix
        old_i = cell_type_id2neuron_id[current_cell_type_index_i][index_of_cell_in_cell_type_i]  
        index_of_cell_in_cell_type_i += 1

        current_cell_type_index_j = 0
        index_of_cell_in_cell_type_j = 0
        for j in range(len(neuron_id_2_cell_type_id)):
            if index_of_cell_in_cell_type_j >= len(cell_type_id2neuron_id[current_cell_type_index_j]):
                current_cell_type_index_j += 1
                index_of_cell_in_cell_type_j = 0

            # get the index of the cell in the original correlation matrix
            old_j = cell_type_id2neuron_id[current_cell_type_index_j][index_of_cell_in_cell_type_j]
            index_of_cell_in_cell_type_j += 1

            connectivity_matrix_new[i, j] = estimated_connectivity_matrix[old_i, old_j]

    return connectivity_matrix_new


# connectivity_matrix_new - grouped N*N connectivity matrix
# cell_type_index2cell_type - dictionary of cell type index to cell type
# cell_type_count - dictionary of cell type to number of cells in that cell type
def calculate_cell_type_level_connectivity_matrix(connectivity_matrix_new, cell_type_id2cell_type, cell_type_count):
    # Calculate the average correlation for each cell type ##################################################################

    connectivity_matrix_cell_type_level = np.zeros((len(cell_type_id2cell_type), len(cell_type_id2cell_type)))

    accumulated_num_cells_i = 0
    for i in range(len(cell_type_id2cell_type)):
        old_i_start = accumulated_num_cells_i
        accumulated_num_cells_i += cell_type_count[cell_type_id2cell_type[i]]

        accumulated_num_cells_j = 0
        for j in range(len(cell_type_id2cell_type)):
            old_j_start = accumulated_num_cells_j
            accumulated_num_cells_j += cell_type_count[cell_type_id2cell_type[j]]

            # (Only for correlation) Remember to correct the denominator to be (total elements - # of cells in each cluster) when calculate for digonal elements
            num_elements_i = accumulated_num_cells_i - old_i_start
            num_elements_j = accumulated_num_cells_j - old_j_start
            total_num_elements = num_elements_i * num_elements_j
            # if i == j:
            #     total_num_elements = total_num_elements - cell_type_count[cell_type_index2cell_type[i]]
            if total_num_elements == 0:
                connectivity_matrix_cell_type_level[i, j] = 0

            connectivity_matrix_cell_type_level[i, j] = np.sum(connectivity_matrix_new[old_i_start : accumulated_num_cells_i, old_j_start : accumulated_num_cells_j]) / total_num_elements

    return connectivity_matrix_cell_type_level


def calculate_cell_type_level_connectivity_matrix_remove_no_connection(connectivity_matrix_new, connectivity_matrix_GT, cell_type_id2cell_type, cell_type_count):
    connectivity_matrix_cell_type_level = np.zeros((len(cell_type_id2cell_type), len(cell_type_id2cell_type)))

    accumulated_num_cells_i = 0
    for i in range(len(cell_type_id2cell_type)):
        old_i_start = accumulated_num_cells_i
        accumulated_num_cells_i += cell_type_count[cell_type_id2cell_type[i]]

        accumulated_num_cells_j = 0
        for j in range(len(cell_type_id2cell_type)):
            old_j_start = accumulated_num_cells_j
            accumulated_num_cells_j += cell_type_count[cell_type_id2cell_type[j]]

            # count the number of non-zeros in the ground truth matrix
            mask_non_zeros = connectivity_matrix_GT[old_i_start : accumulated_num_cells_i, old_j_start : accumulated_num_cells_j] != 0
            total_num_non_zeros_elements = np.sum(mask_non_zeros)

            if total_num_non_zeros_elements == 0:
                connectivity_matrix_cell_type_level[i, j] = 0
            else:
                connectivity_matrix_cell_type_level[i, j] = np.sum(connectivity_matrix_new[old_i_start : accumulated_num_cells_i, old_j_start : accumulated_num_cells_j][mask_non_zeros]) / total_num_non_zeros_elements

    return connectivity_matrix_cell_type_level
            
            



########################################################################################
########################################################################################
## For processing mouse data with multiple sessions.
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

    return all_sessions_new_UniqueID, num_unique_neurons    # shape: (num_sessions, num_neurons_per_session)



def assign_unique_cell_type_ids(all_sessions_original_cell_type, num_neurons_per_session):
    """
    all_sessions_original_cell_type: a concatenated list of the original cell types from all sessions)

    Return:
    all_sessions_new_cell_type: a list of sessions new cell type, each session is a 1D array of shape num_neurons
    """

    unique_cell_types = list(set(all_sessions_original_cell_type))
    unique_cell_types.sort()
    
    # Assign IDs to cell types
    cell_type_order = [unique_cell_types[i] for i in range(len(unique_cell_types))]
    print('cell_type_order:', cell_type_order)   

    # Get the number of neurons in each cell type
    # cell_type_id2number = {cell_type_order[i]: 0 for i in range(len(cell_type_order))}

    # Get new cell type IDs
    all_sessions_new_cell_type_id = np.zeros(len(all_sessions_original_cell_type)).astype(int)
    for i in range(len(all_sessions_original_cell_type)):
        all_sessions_new_cell_type_id[i] = cell_type_order.index(all_sessions_original_cell_type[i])

        # cell_type_id2number[all_sessions_original_cell_type[i]] += 1
    
    # print(cell_type_id2number)

    # Segment all_sessions_new_cell_type_id into sessions
    all_sessions_new_cell_type_id = np.split(all_sessions_new_cell_type_id, np.cumsum(num_neurons_per_session)[:-1])

    return all_sessions_new_cell_type_id, cell_type_order     # shape: num_sessions x num_neurons_per_session



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




########################################################################################
########################################################################################
## For Multi-session mouse data, after getting multi-session N*N connectivity matrices,
## to evaluate result, we need to convert multiple N*N matrices into ONE K*K matrix.
## Works for BOTH connectivity strength AND connectivity probability.
##
## There are 2 ways to do this:
## 1) From multi-session N*N matrices directly get one K*K matrix
## 2) From multi-session N*N matrices get multi-session K*K matrices, then get one K*K matrix
########################################################################################
########################################################################################

def multisession_NN_to_KK_1(
    multisession_NN_list: list,
    multisession_binary_NN_list: list,
    cell_type_order: list,
    multisession_cell_type_id_list: list,
):
    """
    This function can be used for mouse data to compute K*K connectivity strength or K*K connectivity probability
    from multiple sessions N*N results.
    It follows the 1) way to get one K*K matrix directly from multiple N*N matrices.

    - multisession_NN_list: a list of N*N connectivity matrices from multiple sessions 
    (N can be different in different sessions)
    - multisession_binary_NN_list: a list of binary N*N connectivity matrices from multiple sessions
    This is used to represent whether there is a connection or there is no connection.
    If (multisession_binary_NN_list == none), then we won't use it in this function. For computing K*K connection strength,
    it means that we don't first exclude positions where there is no connection. For computing K*K connection probability,
    (multisession_binary_NN_list == none) should be used.
    - cell_type_order: a list of cell type order
    This should contain all cell types that are in the data. (a union of all cell types from all sessions)
    The order should be decided when processing the data. E.g. ['EC', 'Pv', 'Sst', 'Vip']
    - multisession_cell_type_id_list: a list of cell type ids from multiple sessions

    Return:
    KK: a K*K connectivity strength or probability matrix, with cell_type_order as the order of cell types
    """

    KK_result = np.zeros((len(cell_type_order), len(cell_type_order)))
    # count the number of sessions that have non-zero connection between cell type i and cell type j
    KK_count = np.zeros((len(cell_type_order), len(cell_type_order)))

    for i in range(len(multisession_NN_list)):
        current_session_NN = multisession_NN_list[i]
        current_session_cell_type_id = multisession_cell_type_id_list[i]
        if multisession_binary_NN_list is not None:
            current_session_binary_NN = multisession_binary_NN_list[i]

        for j in range(len(current_session_cell_type_id)):
            for k in range(len(current_session_cell_type_id)):

                # Ignore the diagonal elements
                if j == k:
                    continue

                if multisession_binary_NN_list is not None:
                    if current_session_binary_NN[j, k] == 0:
                        continue
                cell_type_j = current_session_cell_type_id[j]
                cell_type_k = current_session_cell_type_id[k]

                KK_result[cell_type_j, cell_type_k] += current_session_NN[j, k]
                KK_count[cell_type_j, cell_type_k] += 1

    # Make all 0 entries in KK_count to be 1, so that we don't divide by 0
    KK_count[KK_count == 0] = 1
    return KK_result / KK_count


def multisession_NN_to_KK_2(
    multisession_NN_list: list,
    multisession_binary_NN_list: list,
    cell_type_order: list,
    multisession_cell_type_id_list: list,
):
    """
    This function can be used for mouse data to compute K*K connectivity strength or K*K connectivity probability
    from multiple sessions N*N results.
    It follows the 2) way to first get multi-session K*K matrices, then get one K*K matrix.

    - multisession_NN_list: a list of N*N connectivity matrices from multiple sessions 
    (N can be different in different sessions)
    - multisession_binary_NN_list: a list of binary N*N connectivity matrices from multiple sessions
    This is used to represent whether there is a connection or there is no connection.
    If (multisession_binary_NN_list == none), then we won't use it in this function. For computing K*K connection strength,
    it means that we don't first exclude positions where there is no connection. For computing K*K connection probability,
    (multisession_binary_NN_list == none) should be used.
    - cell_type_order: a list of cell type order
    This should contain all cell types that are in the data. (a union of all cell types from all sessions)
    The order should be decided when processing the data. E.g. ['EC', 'Pv', 'Sst', 'Vip']
    - multisession_cell_type_id_list: a list of cell type ids from multiple sessions

    Return:
    KK: a K*K connectivity strength or probability matrix, with cell_type_order as the order of cell types
    """

    # First get multi-session K*K matrices
    multisession_KK_list = []
    # This is used to represent whether cell type i and cell type j exist in each session
    multisession_binary_KK_list = []

    for i in range(len(multisession_NN_list)):
        current_session_NN = multisession_NN_list[i]
        current_session_cell_type_id = multisession_cell_type_id_list[i]
        if multisession_binary_NN_list is not None:
            current_session_binary_NN = multisession_binary_NN_list[i]

        current_session_binary_KK = np.zeros((len(cell_type_order), len(cell_type_order)))

        current_session_KK = np.zeros((len(cell_type_order), len(cell_type_order)))
        current_session_KK_count = np.zeros((len(cell_type_order), len(cell_type_order)))
        for j in range(len(current_session_cell_type_id)):
            for k in range(len(current_session_cell_type_id)):
                
                # Ignore the diagonal elements
                if j == k:
                    continue

                if multisession_binary_NN_list is not None:
                    if current_session_binary_NN[j, k] == 0:
                        continue
                cell_type_j = current_session_cell_type_id[j]
                cell_type_k = current_session_cell_type_id[k]

                current_session_KK[cell_type_j, cell_type_k] += current_session_NN[j, k]
                current_session_KK_count[cell_type_j, cell_type_k] += 1

                if current_session_binary_KK[cell_type_j, cell_type_k] == 0:
                    current_session_binary_KK[cell_type_j, cell_type_k] = 1

        # Make all 0 entries in current_session_KK_count to be 1, so that we don't divide by 0
        current_session_KK_count[current_session_KK_count == 0] = 1
        multisession_KK_list.append(current_session_KK / current_session_KK_count)
        multisession_binary_KK_list.append(current_session_binary_KK)

    KK_count = np.sum(multisession_binary_KK_list, axis=0)
    KK_result = np.sum(multisession_KK_list, axis=0)

    return KK_result / KK_count


# This function again can be used for both connectivity strength AND connectivity probability.
def experiment_KK_to_eval_KK(
    experiment_KK: np.ndarray,
    experiment_cell_type_order: list,
    eval_cell_type_order: list,
):
    """
    This function is used to convert experiment_KK to eval_KK, since eval cell type is a subset of experiment cell type.
    The output KK will be a matrix with eval cell type order.
    """

    eval_KK = np.zeros((len(eval_cell_type_order), len(eval_cell_type_order)))
    for eval_cell_type_i in eval_cell_type_order:
        if eval_cell_type_i in experiment_cell_type_order:
            experiment_index_i = experiment_cell_type_order.index(eval_cell_type_i)
            eval_index_i = eval_cell_type_order.index(eval_cell_type_i)

            for eval_cell_type_j in eval_cell_type_order:
                if eval_cell_type_j in experiment_cell_type_order:
                    experiment_index_j = experiment_cell_type_order.index(eval_cell_type_j)
                    eval_index_j = eval_cell_type_order.index(eval_cell_type_j)

                    eval_KK[eval_index_i, eval_index_j] = experiment_KK[experiment_index_i, experiment_index_j]

    return eval_KK