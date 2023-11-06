from math import ceil
import numpy as np
from scipy import stats
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from func2graph.layers import (
    Residual,
    Residual_For_Attention,
    Attention,
    PositionalEncoding,
)



##################
# Baseline_1:
# - Pearson correlation
# - Cross correlation    # TODO
# - Granger Causality      # TODO
##################



def get_activity_correlation_matrix(activity, type="pearson"):
    if type == "pearson":
        correlation_matrix = np.corrcoef(activity) 
    return correlation_matrix




##################
# Baseline_2:
# - Base_2 is the base for Baseline_2
# - Baseline_2 takes in activity from one previous time step to predict for the next time step
##################



class Base_2(pl.LightningModule):
    def __init__(self,) -> None:
        super().__init__()
        self.save_hyperparameters()

    def forward(self, x):
        return NotImplementedError

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=1e-5)

        if self.hparams.scheduler == "plateau":
            lr_scheduler = {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    # TODO: add an argument to control the patience
                    optimizer,
                    patience=6,
                ),
                "monitor": str(self.hparams.loss_function) + " val_loss",
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

        if self.hparams.loss_function == "mse":
            loss = F.mse_loss(y_hat, y, reduction="mean")
        elif self.hparams.loss_function == "poisson":
            if self.hparams.log_input:
                loss = F.poisson_nll_loss(y_hat, x, log_input=True, reduction="mean")
            else:
                loss = F.poisson_nll_loss(y_hat, x, log_input=False, reduction="mean")

        self.log(str(self.hparams.loss_function) + " train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
        if self.hparams.loss_function == "mse":
            loss = F.mse_loss(y_hat, y, reduction="mean")
        elif self.hparams.loss_function == "poisson":
            if self.hparams.log_input:
                loss = F.poisson_nll_loss(y_hat, x, log_input=True, reduction="mean")
            else:
                loss = F.poisson_nll_loss(y_hat, x, log_input=False, reduction="mean")

        result = torch.stack([y_hat.cpu().detach(), y.cpu().detach()], dim=1)

        self.log(str(self.hparams.loss_function) + " val_loss", loss)
        return result

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
        if self.hparams.loss_function == "mse":
            loss = F.mse_loss(y_hat, y, reduction="mean")
        elif self.hparams.loss_function == "poisson":
            if self.hparams.log_input:
                loss = F.poisson_nll_loss(y_hat, x, log_input=True, reduction="mean")
            else:
                loss = F.poisson_nll_loss(y_hat, x, log_input=False, reduction="mean")
    
        self.log(str(self.hparams.loss_function) + " test_loss", loss)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y_hat = self(x)

        return torch.cat((y_hat.cpu().detach(), y.cpu().detach()), dim=1) # batch_size * (2 * neuron_num)

    



# simulated_network_type:
#   1: W_ij @ F.tanh(X_t) + b
#   2: F.tanh(W_ij @ X_t + b)
class Baseline_2(Base_2):
    def __init__(
        self,
        neuron_num=10,
        learning_rate=1e-4,
        simulated_network_type=1,
        model_random_seed=42,
        scheduler="plateau",
        loss_function="mse",             # mse, poisson
        log_input=False,
    ):
        super().__init__()
        self.save_hyperparameters()

        torch.manual_seed(model_random_seed)

        self.simulated_network_type = simulated_network_type

        self.activation = nn.Tanh()
        # self.activation = nn.ELU()

        self.W = nn.Linear(neuron_num, neuron_num, bias=False)
        self.b = nn.Parameter(torch.zeros(neuron_num))
       
    def forward(self, x): # x: batch_size * (neuron_num)
        if self.simulated_network_type == 1:
            x = self.activation(x)
            x = self.W(x) + self.b
        elif self.simulated_network_type == 2:
            x = self.W(x) + self.b
            x = self.activation(x)
        return x