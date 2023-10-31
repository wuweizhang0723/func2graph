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