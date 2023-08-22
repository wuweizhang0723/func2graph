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
def construct_weight_matrix(neuron_num, type='nearest_neighbor'):
    if type=='nearest_neighbor':
        # random sample neuron_num points on 1-D line
        # then construct weight matrix based on the distance between each pair of points
        positions = torch.rand(neuron_num)
        distances = torch.zeros(neuron_num, neuron_num)
        weight_matrix = torch.zeros(neuron_num, neuron_num)
        for i in range(neuron_num):
            for j in range(i+1, neuron_num):
                distances[i, j] = torch.abs(positions[i] - positions[j])
                distances[j, i] = distances[i, j]
                weight_matrix[i, j] = 1 / distances[i, j]
                weight_matrix[j, i] = weight_matrix[i, j]
        
        # normalize weight matrix
        mean = torch.mean(weight_matrix)
        std = torch.std(weight_matrix)
        weight_matrix = (weight_matrix - mean) / std

        # Contol sparsity
        # weight_matrix[weight_matrix < 0.1] = 0
    return weight_matrix



