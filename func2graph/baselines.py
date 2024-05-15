from math import ceil
import numpy as np
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from pyinform import transferentropy, mutualinfo
from func2graph.layers import (
    Residual,
    Residual_For_Attention,
    Attention,
    PositionalEncoding,
)



###################################################################################################
# Baseline_1:
# - Pearson correlation
# - Cross correlation
###################################################################################################


def get_activity_pearson_correlation_matrix(activity):
    correlation_matrix = np.corrcoef(activity)
    # remove diagonal elements
    correlation_matrix = correlation_matrix - np.diag(np.diag(correlation_matrix)) 
    return correlation_matrix


def get_activity_cross_correlation_matrix(activity, tau=1):
    neuron_num = activity.shape[0]
    cross_corr = np.zeros((neuron_num, neuron_num))

    for i in range(neuron_num):
        for j in range(neuron_num):
            postsynaptic = activity[i, tau:]
            if tau == 0:
                presynaptic = activity[j, :]
            else:
                presynaptic = activity[j, :-tau]
            # normalize
            # presynaptic = (presynaptic - np.mean(presynaptic)) / np.std(presynaptic)
            # postsynaptic = (postsynaptic - np.mean(postsynaptic)) / np.std(postsynaptic)
            cross_corr[i, j] = np.corrcoef(presynaptic, postsynaptic)[0, 1]
    return cross_corr




###################################################################################################
# Mutual Information
###################################################################################################

def get_activity_mutual_information(activity, tau=1, bins=10):
    neuron_num = activity.shape[0]
    mutual_information = np.zeros((neuron_num, neuron_num))

    bin_edges = np.linspace(np.min(activity), np.max(activity), bins + 1)

    for i in range(neuron_num):
        for j in range(i, neuron_num):
            if tau == 0:
                presynaptic = activity[j]
                postsynaptic = activity[i]
            else:
                presynaptic = activity[j, :-tau]
                postsynaptic = activity[i, tau:]

            digitized_presynaptic = np.digitize(presynaptic, bins=bin_edges)
            digitized_postsynaptic = np.digitize(postsynaptic, bins=bin_edges)

            # Calculate mutual information
            score = mutualinfo.mutual_info(digitized_presynaptic, digitized_postsynaptic)

            mutual_information[i, j] = score
            mutual_information[j, i] = score

    return mutual_information



###################################################################################################
# Transfer Entropy
###################################################################################################

def get_activity_transfer_entropy(activity, history=1, bins=10):
    """
    Compute the NxN connectivity matrix using transfer entropy.

    Parameters:
    activity (numpy.ndarray): NxT matrix of neural activity.
    history (int): The number of past time steps to consider for each neuron.
    future (int): The number of future time steps to predict.

    Returns:
    numpy.ndarray: NxN matrix of transfer entropy values.
    """
    N, T = activity.shape
    connectivity_matrix = np.zeros((N, N))

    bin_edges = np.linspace(np.min(activity), np.max(activity), bins + 1)

    # Loop over all pairs of neurons
    for source in range(N):
        for target in range(N):
            # if source != target:  # Avoid self-comparison, if desired
            
            # Extract the time series for the source and target
            src_series = activity[source, :]
            tgt_series = activity[target, :]

            digitized_presynaptic = np.digitize(src_series, bins=bin_edges)
            digitized_postsynaptic = np.digitize(tgt_series, bins=bin_edges)

            # Calculate transfer entropy from source to target
            te = transferentropy.transfer_entropy(digitized_presynaptic, digitized_postsynaptic, k=history)

            # Store the transfer entropy in the matrix
            connectivity_matrix[target, source] = te

    return connectivity_matrix




###################################################################################################
# Baseline_2:
# - Base_2 is the base for Baseline_2
# - Baseline_2 takes in activity from one previous time step to predict for the next time step
###################################################################################################


class Base_2(pl.LightningModule):
    def __init__(self,) -> None:
        super().__init__()
        self.save_hyperparameters()

    def forward(self, x):
        return NotImplementedError

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)

        if self.hparams.scheduler == "plateau":
            lr_scheduler = {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    # TODO: add an argument to control the patience
                    optimizer,
                    patience=6,
                ),
                "monitor": "val_loss",
            }
        elif self.hparams.scheduler == "cycle":
            lr_scheduler = {
                "scheduler": torch.optim.lr_scheduler.CyclicLR(
                    optimizer,
                    base_lr=self.hparams.learning_rate / 2,
                    max_lr=self.hparams.learning_rate * 2,
                    cycle_momentum=False,
                ),
                "interval": "step",
            }
        else:
            print("No scheduler is used")

        return [optimizer], [lr_scheduler]
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss = F.mse_loss(y_hat, y, reduction="mean")

        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
        loss = F.mse_loss(y_hat, y, reduction="mean")

        result = torch.stack([y_hat.cpu().detach(), y.cpu().detach()], dim=1)

        self.log("val_loss", loss)
        return result

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
        loss = F.mse_loss(y_hat, y, reduction="mean")
    
        self.log("test_loss", loss)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y_hat = self(x)

        return torch.cat((y_hat.cpu().detach(), y.cpu().detach()), dim=1) # batch_size * (2 * neuron_num)

    
#   tanh(W_ij @ X_t + b)
#   exp(W_ij @ X_t + b)
class GLM_sim(Base_2):
    def __init__(
        self,
        neuron_num=10,
        learning_rate=1e-4,
        model_random_seed=42,
        scheduler="cycle",
        loss_function="mse",   # mse
        weight_decay=0,
        activation_type="tanh",   # tanh, exp, sigmoid
    ):
        super().__init__()
        self.save_hyperparameters()

        torch.manual_seed(model_random_seed)

        self.activation_type = activation_type
        if activation_type == "tanh":
            self.activation = nn.Tanh()

        self.W = nn.Linear(neuron_num, neuron_num, bias=False)
        self.b = nn.Parameter(torch.zeros(neuron_num))
       
    def forward(self, x): # x: batch_size * (neuron_num)
        x = self.W(x) + self.b

        if self.activation_type == "tanh":
            x = self.activation(x)
        elif self.activation_type == "exp":
            x = torch.exp(x)
        elif self.activation_type == "sigmoid":
            x = torch.sigmoid(x)
        return x
    



###################################################################################################
# GLM_M: GLM for real mouse data
# - Base_3 is the base for GLM_M
# - x_{t+1} = \sum A_k x_{t-k}, where k is the number of tau(s)
###################################################################################################

class Base_3(pl.LightningModule):
    def __init__(self,) -> None:
        super().__init__()
        self.save_hyperparameters()

    def forward(self, x):
        return NotImplementedError
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)

        if self.hparams.scheduler == "plateau":
            lr_scheduler = {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    # TODO: add an argument to control the patience
                    optimizer,
                    patience=3,
                ),
                "monitor": "VAL_mse_loss",
            }
        else:
            print("No scheduler is used")

        return [optimizer], [lr_scheduler]
    
    def training_step(self, batch, batch_idx):
        x = batch[0]  # x: batch_size * (neuron_num * tau)

        # Make the last time step as the target
        target = x[:, :, -1].clone()
        pred = self(x[:, :, :-1])

        # make pred and target have the same shape
        target = target.reshape(pred.shape)

        loss = F.mse_loss(pred, target, reduction="mean")
        self.log("TRAIN_mse_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch[0]

        # Make the last time step as the target
        target = x[:, :, -1].clone()
        pred = self(x[:, :, :-1])

        # make pred and target have the same shape
        target = target.reshape(pred.shape)

        loss = F.mse_loss(pred, target, reduction="mean")
        self.log("VAL_mse_loss", loss)
        return loss
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x = batch[0]

        # Make the last time step as the target
        target = x[:, :, -1].clone()
        pred = self(x[:, :, :-1])

        # make pred and target have the same shape
        target = target.reshape(pred.shape)

        return torch.cat((pred.cpu().detach(), target.cpu().detach()), dim=1)
    

# x_{t+1} = Tanh( \sum A_k x_{t-k} ), Exp( \sum A_k x_{t-k} ), where k is the number of tau(s)
class GLM_M(Base_3):
    def __init__(
        self,
        num_neurons,
        k=1,   # k determines the number of weight matrices in the model
        learning_rate=1e-4,
        scheduler="plateau",
        weight_decay=0,
        model_random_seed=42,
        activation_type="tanh",   # tanh, exp
    ):
        super().__init__()
        self.save_hyperparameters()

        torch.manual_seed(model_random_seed)

        self.W_list = nn.ModuleList()
        self.k = k

        for i in range(k):
            self.W_list.append(nn.Linear(num_neurons, num_neurons, bias=False))   # let's define the order is A_1, A_2, ..., A_k

        self.b = nn.Parameter(torch.randn(num_neurons))

        self.activation_type = activation_type
        if activation_type == "tanh":
            self.activation = nn.Tanh()
       
    def forward(self, x): # x: batch_size * (neuron_num * tau)
        for i in range(1, self.k+1):
            input = x[:, :, -i]

            if i == 1:
                output = self.W_list[i-1](input)
            else:
                output += self.W_list[i-1](input)

        if self.activation_type == "tanh":
            return self.activation(output + self.b)
        elif self.activation_type == "exp":
            return torch.exp(output + self.b)