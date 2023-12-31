from math import ceil
import numpy as np
from scipy import stats
import einops
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
    Spatial_Temporal_Attention,
)



class Base(pl.LightningModule):
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
            return [optimizer]

        return [optimizer], [lr_scheduler]
    
    def training_step(self, batch, batch_idx):
        if self.hparams.task_type == "reconstruction":
            x = batch[0]   # batch_size * (neuron_num*time)
            x_hat = self(x)

            pred = x_hat
            target = x

        elif self.hparams.task_type == "prediction":
            x, y = batch
            y_hat = self(x)
            
            pred = y_hat
            target = y

        elif self.hparams.task_type == "mask":
            # x, y, mask_indices = batch
            # x_hat = self(x)

            # # get the masked part from x_hat to be y_hat
            # y_hat = torch.zeros(y.shape).to(y.device)
            # for i in range(mask_indices.shape[0]):
            #     for j in range(mask_indices.shape[1]):
            #         y_hat[i, :, j] = x_hat[i, :, mask_indices[i, j]]
            #         # y_hat[i, j, :] = x_hat[i, mask_indices[i, j], :]

            x, y, mask_indices_i, mask_indices_j = batch
            x_hat = self(x)

            # get the masked part from x_hat to be y_hat
            y_hat = torch.zeros(y.shape).to(y.device)
            n_sample, mask_size = mask_indices_i.shape
            for i in range(n_sample):
                for j in range(mask_size):
                    y_hat[i, j] = x_hat[i, mask_indices_i[i, j], mask_indices_j[i, j]]

            pred = y_hat
            target = y

        if self.hparams.loss_function == "mse":
            loss = F.mse_loss(pred, target, reduction="mean")
        elif self.hparams.loss_function == "poisson":
            loss = F.poisson_nll_loss(pred, target, log_input=self.hparams.log_input, reduction="mean")
        elif self.hparams.loss_function == "gaussian":
            var = torch.ones(pred.shape, requires_grad=True).to(pred.device)  ##############################
            loss = F.gaussian_nll_loss(pred, target, reduction="mean", var=var)

        self.log(str(self.hparams.loss_function) + " train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.hparams.task_type == "reconstruction":
            x = batch[0]   # batch_size * (neuron_num*time)
            x_hat = self(x)
            
            pred = x_hat
            target = x

            result = torch.stack([x_hat.cpu().detach(), x.cpu().detach()], dim=1)
        elif self.hparams.task_type == "prediction":
            x, y = batch
            y_hat = self(x)
            
            pred = y_hat
            target = y

            result = torch.stack([y_hat.cpu().detach(), y.cpu().detach()], dim=1)

        elif self.hparams.task_type == "mask":
            # x, y, mask_indices = batch
            # x_hat = self(x)

            # # get the masked part from x_hat to be y_hat
            # y_hat = torch.zeros(y.shape).to(y.device)
            # for i in range(mask_indices.shape[0]):
            #     for j in range(mask_indices.shape[1]):
            #         y_hat[i, :, j] = x_hat[i, :, mask_indices[i, j]]
            #         # y_hat[i, j, :] = x_hat[i, mask_indices[i, j], :]

            x, y, mask_indices_i, mask_indices_j = batch
            x_hat = self(x)

            # get the masked part from x_hat to be y_hat
            y_hat = torch.zeros(y.shape).to(y.device)
            n_sample, mask_size = mask_indices_i.shape
            for i in range(n_sample):
                for j in range(mask_size):
                    y_hat[i, j] = x_hat[i, mask_indices_i[i, j], mask_indices_j[i, j]]

            pred = y_hat
            target = y

            result = torch.stack([y_hat.cpu().detach(), y.cpu().detach()], dim=1)

        if self.hparams.loss_function == "mse":
            loss = F.mse_loss(pred, target, reduction="mean")
        elif self.hparams.loss_function == "poisson":
            loss = F.poisson_nll_loss(pred, target, log_input=self.hparams.log_input, reduction="mean")
        elif self.hparams.loss_function == "gaussian":
            var = torch.ones(pred.shape, requires_grad=True).to(pred.device)  ##############################
            loss = F.gaussian_nll_loss(pred, target, reduction="mean", var=var)

        self.log(str(self.hparams.loss_function) + " val_loss", loss)
        return result
    
    # def on_validation_epoch_end(self, validation_step_outputs):
    #     all_val_result = torch.cat(validation_step_outputs, dim=0)

    #     val_corr = stats.pearsonr(
    #         all_val_result[:, 0].flatten(), all_val_result[:, 1].flatten()
    #     ).statistic

    #     self.log("val_corr", val_corr)
    
    def test_step(self, batch, batch_idx):
        if self.hparams.task_type == "reconstruction":
            x = batch[0]   # batch_size * (neuron_num*time)
            x_hat = self(x)
            
            pred = x_hat
            target = x
        elif self.hparams.task_type == "prediction":
            x, y = batch
            y_hat = self(x)
            
            pred = y_hat
            target = y
        elif self.hparams.task_type == "mask":
            # x, y, mask_indices = batch
            # x_hat = self(x)

            # # get the masked part from x_hat to be y_hat
            # y_hat = torch.zeros(y.shape).to(y.device)
            # for i in range(mask_indices.shape[0]):
            #     for j in range(mask_indices.shape[1]):
            #         y_hat[i, :, j] = x_hat[i, :, mask_indices[i, j]]
            #         # y_hat[i, j, :] = x_hat[i, mask_indices[i, j], :]

            x, y, mask_indices_i, mask_indices_j = batch
            x_hat = self(x)

            # get the masked part from x_hat to be y_hat
            y_hat = torch.zeros(y.shape).to(y.device)
            n_sample, mask_size = mask_indices_i.shape
            for i in range(n_sample):
                for j in range(mask_size):
                    y_hat[i, j] = x_hat[i, mask_indices_i[i, j], mask_indices_j[i, j]]

            pred = y_hat
            target = y

        if self.hparams.loss_function == "mse":
            loss = F.mse_loss(pred, target, reduction="mean")
        elif self.hparams.loss_function == "poisson":
            loss = F.poisson_nll_loss(pred, target, log_input=self.hparams.log_input, reduction="mean")
        elif self.hparams.loss_function == "gaussian":
            var = torch.ones(pred.shape, requires_grad=True).to(pred.device)  ##############################
            loss = F.gaussian_nll_loss(pred, target, reduction="mean", var=var)
        
        self.log(str(self.hparams.loss_function) + " test_loss", loss)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        if self.hparams.task_type == "reconstruction":
            x = batch[0]   # batch_size * (neuron_num*time)
            x_hat, attention = self(x)

            return x_hat, x, attention
        
        elif self.hparams.task_type == "prediction":
            x, y = batch
            y_hat, attention = self(x)

            return y_hat, y, attention
        
        elif self.hparams.task_type == "mask":
            # x, y, mask_indices = batch
            # x_hat, attention = self(x)

            # # get the masked part from x_hat to be y_hat
            # y_hat = torch.zeros(y.shape).to(y.device)
            # for i in range(mask_indices.shape[0]):
            #     for j in range(mask_indices.shape[1]):
            #         y_hat[i, :, j] = x_hat[i, :, mask_indices[i, j]]
            #         # y_hat[i, j, :] = x_hat[i, mask_indices[i, j], :]

            x, y, mask_indices_i, mask_indices_j = batch
            x_hat, attention = self(x)

            # get the masked part from x_hat to be y_hat
            y_hat = torch.zeros(y.shape).to(y.device)
            n_sample, mask_size = mask_indices_i.shape
            for i in range(n_sample):
                for j in range(mask_size):
                    y_hat[i, j] = x_hat[i, mask_indices_i[i, j], mask_indices_j[i, j]]

            return y_hat, y, attention
    





class Attention_Autoencoder(Base):
    def __init__(
        self,
        model_random_seed=42,
        neuron_num=10,
        window_size=200,
        hidden_size_1=128, # MLP_1
        h_layers_1=2,
        heads=1,  # Attention
        attention_layers=1,
        dim_key=64,
        hidden_size_2=256, # MLP_2
        h_layers_2=2,
        dropout=0.2,
        learning_rate=1e-4,
        scheduler="plateau",
        prediction_mode=False,
        pos_enc_type="none",  # "sin_cos" or "lookup_table" or "none"
        task_type = "reconstruction",    # "reconstruction" or "prediction" or "mask"
        predict_window_size = 100,
        loss_function = "mse", # "mse" or "poisson" or "gaussian"
        log_input = False,
        attention_activation = "softmax", # "softmax" or "sigmoid" or "tanh"
        weight_decay = 0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.prediction_mode = prediction_mode

        torch.manual_seed(model_random_seed)

        # MLP_1

        if (task_type == "reconstruction") or (task_type == "mask"):
            hidden_size_1 = window_size
        elif task_type == "prediction":
            hidden_size_1 = window_size - predict_window_size


        # if task_type == "reconstruction":
        #     hidden_size_1 = window_size

        #     self.fc1 = nn.Sequential(
        #         nn.Linear(window_size, hidden_size_1), nn.ReLU()
        #     )
        # elif task_type == "prediction":
        #     hidden_size_1 = window_size - predict_window_size       #######################

        #     self.fc1 = nn.Sequential(
        #         nn.Linear(window_size - predict_window_size, hidden_size_1), nn.ReLU()
        #     )

        # self.fclayers1 = nn.ModuleList(
        #     nn.Sequential(
        #         nn.Linear(hidden_size_1, hidden_size_1), nn.ReLU(), nn.Dropout(dropout)
        #     )
        #     for layer in range(h_layers_1)
        # )


        # Attention

        self.pos_enc_type = pos_enc_type
        if pos_enc_type == "sin_cos":
            self.position_enc = PositionalEncoding(hidden_size_1, neuron_num=neuron_num)
            dim_in = hidden_size_1
            self.layer_norm = nn.LayerNorm(dim_in)
        elif pos_enc_type == "lookup_table":
            self.embedding_table = nn.Embedding(
                num_embeddings=neuron_num, embedding_dim=hidden_size_1
            )
            dim_in = hidden_size_1
            self.layer_norm = nn.LayerNorm(dim_in)
        else:
            dim_in = hidden_size_1

        self.attentionlayers = nn.ModuleList()

        for layer in range(attention_layers):
            # self.attentionlayers.append(
            #     nn.Sequential(
            #         Residual_For_Attention(
            #             Attention(
            #                 dim=dim_in,  # the last dimension of input
            #                 heads=heads,
            #                 prediction_mode=self.prediction_mode,
            #             ),
            #             prediction_mode=self.prediction_mode,
            #         ),
            #     )
            # )
            self.attentionlayers.append(
                nn.Sequential(
                    Attention(
                        dim=dim_in,  # the last dimension of input
                        heads=heads,
                        dim_key=dim_key,
                        prediction_mode=self.prediction_mode,
                        activation = attention_activation,
                    ),
                )
            )
            self.attentionlayers.append(
                nn.Sequential(
                    nn.LayerNorm(dim_in),
                    Residual(
                        nn.Sequential(
                            nn.Linear(dim_in, hidden_size_1 * 2),
                            nn.Dropout(dropout),
                            nn.ReLU(),
                            nn.Linear(hidden_size_1 * 2, dim_in),
                            nn.Dropout(dropout),
                            nn.ReLU(),
                        )
                    ),
                    nn.LayerNorm(dim_in),
                )
            )

        # MLP_2

        self.fc2 = nn.Sequential(
            nn.Linear(dim_in, hidden_size_2), nn.ReLU()
        )

        self.fclayers2 = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden_size_2, hidden_size_2), nn.ReLU(), nn.Dropout(dropout)
            )
            for layer in range(h_layers_2)
        )

        if (task_type == "reconstruction") or (task_type == "mask"):
            self.out = nn.Linear(hidden_size_2, window_size)
        elif task_type == "prediction":
            self.out = nn.Linear(hidden_size_2, predict_window_size)


    def forward(self, x): # x: batch_size * (neuron_num*time)
        # x = self.fc1(x)
        # for layer in self.fclayers1:
        #     x = layer(x)

        if self.pos_enc_type == "sin_cos":
            # Add positional encoding
            x = self.position_enc(x)
            x = self.layer_norm(x)
        elif self.pos_enc_type == "lookup_table":
            # Add positional encoding
            idx = torch.arange(x.shape[1]).to(x.device)
            x = x + self.embedding_table(idx)
            # embedding = einops.repeat(self.embedding_table(idx), 'n d -> b n d', b=x.shape[0])
            # x = torch.concat([x, embedding], dim=-1)
            x = self.layer_norm(x)

        attention_results = []
        print('length: ', len(self.attentionlayers))
        for layer in self.attentionlayers:
            x = layer(x)
            if type(x) is tuple:
                print('3')
                x, attn = x
                attention_results.append(attn)

        x = self.fc2(x)
        for layer in self.fclayers2:
            x = layer(x)

        x = self.out(x)

        if self.hparams.prediction_mode == True:
            return x, attention_results[0]
        else:
            return x








# spatial_temporal_1: spatial attention has no value matrix
# spatial_temporal_2: temporal attention has no value matrix
# spatial_temporal_3: spatial attention has value matrix, temporal attention has value matrix
# spatial: there is no temporal attention
#
class Spatial_Temporal_Attention_Model(Base):
    def __init__(
        self,
        model_random_seed=42,
        neuron_num=10,
        window_size=200,
        predict_window_size = 100,
        hidden_size_1_T=128, # MLP_1
        hidden_size_1_S=128,
        h_layers_1=2,
        attention_type = "spatial_temporal_1", # "spatial_temporal_1" or "spatial_temporal_2" or "spatial_temporal _3" or "spatial"
        pos_enc_type="none",  # "sin_cos" or "lookup_table" or "none"
        heads=1,  # Attention
        hidden_size_2=256, # MLP_2
        h_layers_2=2,
        dropout=0.2,
        learning_rate=1e-4,
        prediction_mode=False,
        task_type = "prediction",    # "reconstruction" or "prediction"
    ):
        super().__init__()
        self.save_hyperparameters()

        self.prediction_mode = prediction_mode

        torch.manual_seed(model_random_seed)

        # MLP_1

        # x_S
        if task_type == "reconstruction":
            self.fc1_S = nn.Sequential(
                nn.Linear(window_size, hidden_size_1_S), nn.ReLU()
            )
            # dim_S = T, dim_T = N
            T = window_size
            N = neuron_num
        elif task_type == "prediction":
            self.fc1_S = nn.Sequential(
                nn.Linear(window_size - predict_window_size, hidden_size_1_S), nn.ReLU()
            )
            # dim_S = T, dim_T = N
            T = (window_size-predict_window_size)
            N = neuron_num

        # x_T
        self.fc1_T = nn.Sequential(
            nn.Linear(neuron_num, hidden_size_1_T), nn.ReLU()
        )

        self.fclayers1_S = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden_size_1_S, hidden_size_1_S), nn.ReLU(), nn.Dropout(dropout)
            )
            for layer in range(h_layers_1)
        )

        self.fclayers1_T = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden_size_1_T, hidden_size_1_T), nn.ReLU(), nn.Dropout(dropout)
            )
            for layer in range(h_layers_1)
        )

        if task_type == "reconstruction":
            self.fclayers1_S.append(
                nn.Sequential(
                    nn.Linear(hidden_size_1_S, window_size), nn.ReLU()
                )
            )
        elif task_type == "prediction":
            self.fclayers1_S.append(
                nn.Sequential(
                    nn.Linear(hidden_size_1_S, window_size - predict_window_size), nn.ReLU()
                )
            )
        
        self.fclayers1_T.append(
            nn.Sequential(
                nn.Linear(hidden_size_1_T, neuron_num), nn.ReLU()
            )
        )


        # Positional Encoding

        self.pos_enc_type = pos_enc_type
        if pos_enc_type == "sin_cos":
            # X_S
            self.position_enc_S = PositionalEncoding(encoding_dim=T, num=N)
            self.layer_norm_S = nn.LayerNorm(T)
            # X_T
            self.position_enc_T = PositionalEncoding(encoding_dim=N, num=T)
            self.layer_norm_T = nn.LayerNorm(N)
        elif pos_enc_type == "lookup_table":
            # X_S
            self.embedding_table_S = nn.Embedding(num_embeddings=N, embedding_dim=T)
            self.layer_norm_S = nn.LayerNorm(T)
            # X_T
            self.embedding_table_T = nn.Embedding(num_embeddings=T, embedding_dim=N)
            self.layer_norm_T = nn.LayerNorm(N)

        # Attention

        self.attentionlayers = nn.ModuleList()

        # self.attentionlayers.append(
        #     nn.Sequential(
        #         Residual_For_Attention(
        #             Spatial_Temporal_Attention(
        #                 dim_T = N,
        #                 dim_S = T,
        #                 heads=heads,
        #                 dim_key_T=N,
        #                 dim_value_T=N,    # dim_value_T = dim_T = N must be satisfied
        #                 dim_key_S=T,
        #                 dim_value_S=T,   # dim_value_S = dim_S = T must be satisfied
        #                 prediction_mode=self.prediction_mode,
        #                 attention_type = attention_type,
        #                 pos_enc_type=pos_enc_type,
        #             ),
        #             prediction_mode=self.prediction_mode,
        #         ),
        #     )
        # )

        self.attentionlayers.append(
            nn.Sequential(
                Spatial_Temporal_Attention(
                    dim_T = N,
                    dim_S = T,
                    heads=heads,
                    dim_key_T=N,
                    dim_value_T=N,    # dim_value_T = dim_T = N must be satisfied
                    dim_key_S=T,
                    dim_value_S=T,   # dim_value_S = dim_S = T must be satisfied
                    prediction_mode=self.prediction_mode,
                    attention_type = attention_type,
                    pos_enc_type=pos_enc_type,
                ),
            )
        )

        self.attentionlayers.append(
            nn.Sequential(
                nn.LayerNorm(T),
                Residual(
                    nn.Sequential(
                        nn.Linear(T, T * 2),
                        nn.Dropout(dropout),
                        nn.ReLU(),
                        nn.Linear(T * 2, T),
                        nn.Dropout(dropout),
                    )
                ),
                nn.LayerNorm(T),
            )
        )

        # MLP

        self.fc2 = nn.Sequential(
            nn.Linear(T, hidden_size_2), nn.ReLU()
        )

        self.fclayers2 = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden_size_2, hidden_size_2), nn.ReLU(), nn.Dropout(dropout)
            )
            for layer in range(h_layers_2)
        )

        if task_type == "reconstruction":
            self.out = nn.Linear(hidden_size_2, window_size)
        elif task_type == "prediction":
            self.out = nn.Linear(hidden_size_2, predict_window_size)

    def forward(self, x): # x: batch_size * neuron_num * time
        x_S = x.clone()
        x_T = x.permute(0, 2, 1)

        # x_S
        x_S = self.fc1_S(x_S)
        for layer in self.fclayers1_S:
            x_S = layer(x_S)

        # x_T
        x_T = self.fc1_T(x_T)
        for layer in self.fclayers1_T:
            x_T = layer(x_T)

        # Positional Encoding

        if self.pos_enc_type == "sin_cos":
            # Add positional encoding
            x_T = self.position_enc_T(x_T)
            x_T = self.layer_norm_T(x_T)

            x_S = self.position_enc_S(x_S)
            x_S = self.layer_norm_S(x_S)
        elif self.pos_enc_type == "lookup_table":
            # Add positional encoding
            idx = torch.arange(x_T.shape[1]).to(x_T.device)
            x_T = x_T + self.embedding_table_T(idx)
            x_T = self.layer_norm_T(x_T)

            idx = torch.arange(x_S.shape[1]).to(x_S.device)
            x_S = x_S + self.embedding_table_S(idx)
            x_S = self.layer_norm_S(x_S)

        attention_results = []

        x = (x_T, x_S)
        x = self.attentionlayers[0](x)
        if type(x) is tuple:
            x, attn = x
            attention_results.append(attn)

        x = self.attentionlayers[1](x)

        # MLP_2

        x = self.fc2(x)
        for layer in self.fclayers2:
            x = layer(x)

        x = self.out(x)

        if self.hparams.prediction_mode == True:
            return x, attention_results[0]
        else:
            return x

        




##############################################################################################################
##############################################################################################################
## For Attention with Constraint Model
##############################################################################################################
##############################################################################################################

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
                    patience=3,
                ),
                "monitor": "VAL_sum_loss",
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
        print("self.cell_type_level_constraint gradient: ", self.cell_type_level_constraint.grad)
        print("self.cell_type_level_constraint: ", self.cell_type_level_constraint)

        x, neuron_ids, cell_type_ids = batch         # x is entire window
        print("unique num cell types: ", np.unique(cell_type_ids[0].clone().detach().cpu().numpy()))
        x = x.squeeze(0)                 # remove the fake batch_size
        neuron_ids = neuron_ids.squeeze(0)
        cell_type_ids = cell_type_ids.squeeze(0)

        # Make the last time step as the target
        target = x[:, :, -1*self.hparams.predict_window_size:].clone()
        pred, neuron_level_attention, cell_type_level_constraint = self(x[:, :, :-1*self.hparams.predict_window_size], neuron_ids)


        cell_type_ids_np = cell_type_ids[0].clone().detach().cpu().numpy()
        expanded_cell_type_level_constraint = torch.zeros((neuron_level_attention.shape[1],neuron_level_attention.shape[2]), requires_grad=True).to(pred.device)
        # loop over unique cell types

        for i in list(np.unique(cell_type_ids_np)):
            # find the neurons with the same cell type
            neuron_ids_with_same_cell_type_i = np.where(cell_type_ids_np == i)[0]
            for j in list(np.unique(cell_type_ids_np)):
                # find the neurons with the same cell type
                neuron_ids_with_same_cell_type_j = np.where(cell_type_ids_np == j)[0]
                # Assign the same constraint to the neurons with the same cell type
                expanded_cell_type_level_constraint[neuron_ids_with_same_cell_type_i, neuron_ids_with_same_cell_type_j.reshape(-1,1)] = cell_type_level_constraint[i, j]
                expanded_cell_type_level_constraint[neuron_ids_with_same_cell_type_j, neuron_ids_with_same_cell_type_i.reshape(-1,1)] = cell_type_level_constraint[j, i]

    
        # print('num zeros in expanded_cell_type_level_constraint: ', torch.sum(expanded_cell_type_level_constraint == 0))
    
        # Expand the first dimension of expanded_cell_type_level_constraint to batch_size
        expanded_cell_type_level_constraint = einops.repeat(expanded_cell_type_level_constraint, 'n d -> b n d', b=neuron_level_attention.shape[0])
        # print('eeeee', expanded_cell_type_level_constraint.requires_grad)
        # print('kkkkkk', cell_type_level_constraint.requires_grad)
        # print('aaaaaaa', neuron_level_attention.requires_grad)

        # Use Gaussian NLL Loss to add constraint, var should be a hyperparameter
        var_constraint = torch.ones(neuron_level_attention.shape, requires_grad=True).to(pred.device) * self.hparams.constraint_var
        constraint_loss = F.gaussian_nll_loss(neuron_level_attention, expanded_cell_type_level_constraint, reduction="mean", var=var_constraint)
        
        # make pred and target have the same shape
        target = target.reshape(pred.shape)

        if self.hparams.loss_function == "mse":
            loss = F.mse_loss(pred, target, reduction="mean")
        elif self.hparams.loss_function == "poisson":
            loss = F.poisson_nll_loss(pred, target, log_input=self.hparams.log_input, reduction="mean")
        elif self.hparams.loss_function == "gaussian":
            var = torch.ones(pred.shape, requires_grad=True).to(pred.device)  ##############################
            loss = F.gaussian_nll_loss(pred, target, reduction="mean", var=var)

        self.log("TRAIN_constraint_loss", constraint_loss * self.hparams.constraint_loss_weight)
        self.log("TRAIN_" + str(self.hparams.loss_function) + "_loss", loss)
        self.log("TRAIN_sum_loss", loss + constraint_loss * self.hparams.constraint_loss_weight)
        return loss + constraint_loss * self.hparams.constraint_loss_weight
    
    def validation_step(self, batch, batch_idx):

        x, neuron_ids, cell_type_ids = batch         # x is entire window
        x = x.squeeze(0)                 # remove the fake batch_size
        neuron_ids = neuron_ids.squeeze(0)
        cell_type_ids = cell_type_ids.squeeze(0)

        # Make the last time step as the target
        target = x[:, :, -1*self.hparams.predict_window_size:].clone()
        pred, neuron_level_attention, cell_type_level_constraint = self(x[:, :, :-1*self.hparams.predict_window_size], neuron_ids)


        cell_type_ids_np = cell_type_ids[0].clone().detach().cpu().numpy()
        expanded_cell_type_level_constraint = torch.zeros((neuron_level_attention.shape[1],neuron_level_attention.shape[2]), requires_grad=True).to(pred.device)
        # loop over unique cell types

        for i in list(np.unique(cell_type_ids_np)):
            # find the neurons with the same cell type
            neuron_ids_with_same_cell_type_i = np.where(cell_type_ids_np == i)[0]
            for j in list(np.unique(cell_type_ids_np)):
                # find the neurons with the same cell type
                neuron_ids_with_same_cell_type_j = np.where(cell_type_ids_np == j)[0]
                # Assign the same constraint to the neurons with the same cell type
                expanded_cell_type_level_constraint[neuron_ids_with_same_cell_type_i, neuron_ids_with_same_cell_type_j.reshape(-1,1)] = cell_type_level_constraint[i, j]
                expanded_cell_type_level_constraint[neuron_ids_with_same_cell_type_j, neuron_ids_with_same_cell_type_i.reshape(-1,1)] = cell_type_level_constraint[j, i]

            
        expanded_cell_type_level_constraint = einops.repeat(expanded_cell_type_level_constraint, 'n d -> b n d', b=neuron_level_attention.shape[0])

        var_constraint = torch.ones(neuron_level_attention.shape, requires_grad=True).to(pred.device) * self.hparams.constraint_var
        constraint_loss = F.gaussian_nll_loss(neuron_level_attention, expanded_cell_type_level_constraint, reduction="mean", var=var_constraint)
        
        if self.hparams.loss_function == "mse":
            loss = F.mse_loss(pred, target, reduction="mean")
        elif self.hparams.loss_function == "poisson":
            loss = F.poisson_nll_loss(pred, target, log_input=self.hparams.log_input, reduction="mean")
        elif self.hparams.loss_function == "gaussian":
            var = torch.ones(pred.shape, requires_grad=True).to(pred.device)  ##############################
            loss = F.gaussian_nll_loss(pred, target, reduction="mean", var=var)

        self.log("VAL_constraint_loss", constraint_loss * self.hparams.constraint_loss_weight)
        self.log("VAL_" + str(self.hparams.loss_function) + "_loss", loss)
        self.log("VAL_sum_loss", loss + constraint_loss * self.hparams.constraint_loss_weight)
        return loss + constraint_loss * self.hparams.constraint_loss_weight
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, neuron_ids, cell_type_ids = batch         # x is entire window
        x = x.squeeze(0)                 # remove the fake batch_size
        neuron_ids = neuron_ids.squeeze(0)
        cell_type_ids = cell_type_ids.squeeze(0)

        # Make the last time step as the target
        target = x[:, :, -1*self.hparams.predict_window_size:].clone()
        pred, neuron_level_attention, cell_type_level_constraint = self(x[:, :, :-1*self.hparams.predict_window_size], neuron_ids)

        return pred, target, neuron_level_attention
    


class Attention_With_Constraint(Base_2):
    def __init__(
        self,
        num_unqiue_neurons,
        num_cell_types,
        model_random_seed=42,
        window_size=200,
        hidden_size_1=128, # MLP_1
        h_layers_1=2,
        heads=1,  # Attention
        attention_layers=1,
        dim_key=64,
        hidden_size_2=256, # MLP_2
        h_layers_2=2,
        dropout=0.2,
        learning_rate=1e-4,
        scheduler="cycle",
        predict_window_size = 1,
        loss_function = "mse", # "mse" or "poisson" or "gaussian"
        log_input = False,
        constraint_loss_weight = 1,
        attention_activation = "none", # "softmax" or "sigmoid" or "tanh"
        weight_decay = 0,
        constraint_var = 1,
    ):
        super().__init__()
        self.save_hyperparameters()

        torch.manual_seed(model_random_seed)

        # k * k matrix constraint
        # self.cell_type_level_constraint = nn.Parameter(torch.zeros((num_cell_types, num_cell_types)), requires_grad=True)
        # self.cell_type_level_constraint = nn.Parameter(torch.FloatTensor(num_cell_types, num_cell_types).uniform_(-1, 1))
        self.cell_type_level_constraint = nn.Parameter(torch.FloatTensor(num_cell_types, num_cell_types).uniform_(0, 1))

        # MLP_1

        hidden_size_1 = window_size - predict_window_size

        # Attention

        self.embedding_table = nn.Embedding(
            num_embeddings=num_unqiue_neurons, embedding_dim=hidden_size_1   # global unique neuron lookup table
        )
        dim_in = hidden_size_1
        self.layer_norm = nn.LayerNorm(dim_in)

        self.attentionlayers = nn.ModuleList()

        for layer in range(attention_layers):
            self.attentionlayers.append(
                nn.Sequential(
                    Attention(
                        dim=dim_in,  # the last dimension of input
                        heads=heads,
                        dim_key=dim_key,
                        prediction_mode=True,  ##############################
                        activation = attention_activation,
                    ),
                )
            )
            self.attentionlayers.append(
                nn.Sequential(
                    nn.LayerNorm(dim_in),
                    Residual(
                        nn.Sequential(
                            nn.Linear(dim_in, hidden_size_1 * 2),
                            nn.Dropout(dropout),
                            nn.ReLU(),
                            nn.Linear(hidden_size_1 * 2, dim_in),
                            nn.Dropout(dropout),
                            nn.ReLU(),
                        )
                    ),
                    nn.LayerNorm(dim_in),
                )
            )

        # MLP_2

        self.fc2 = nn.Sequential(
            nn.Linear(dim_in, hidden_size_2), nn.ReLU()
        )

        self.fclayers2 = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden_size_2, hidden_size_2), nn.ReLU(), nn.Dropout(dropout)
            )
            for layer in range(h_layers_2)
        )

        self.out = nn.Linear(hidden_size_2, predict_window_size)


    def forward(self, x, neuron_ids): # x: batch_size * (neuron_num*time), neuron_ids: batch_size * neuron_num
        # Add positional encoding
        print("x.shape: ", x.shape)
        print("self.embedding_table(neuron_ids[0]).shape: ", self.embedding_table(neuron_ids[0]).shape)
        x = x + self.embedding_table(neuron_ids[0])   # the first dimension doesn't matter
        x = self.layer_norm(x)

        attention_results = []
        for layer in self.attentionlayers:
            x = layer(x)
            if type(x) is tuple:
                x, attn = x
                attention_results.append(attn)

        x = self.fc2(x)
        for layer in self.fclayers2:
            x = layer(x)

        x = self.out(x)

        batch_neuron_num = attention_results[0].shape[-1]
        return x, attention_results[0].view(-1, batch_neuron_num, batch_neuron_num), self.cell_type_level_constraint
    



##############################################################################################################
##############################################################################################################
## For Attention with Constraint Model (simulated data)
##############################################################################################################
##############################################################################################################

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
                "monitor": "VAL_sum_loss",
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
        y_hat, neuron_level_attention, cell_type_level_constraint = self(x)

        expanded_cell_type_level_constraint = torch.zeros((neuron_level_attention.shape[1],neuron_level_attention.shape[2]), requires_grad=True).to(y_hat.device)
        cell_type_count = [self.neuron_num*0.76, self.neuron_num*0.08, self.neuron_num*0.08, self.neuron_num*0.08]
        # loop over unique cell types
        for i in range(cell_type_level_constraint.shape[0]):
            start = int(sum(cell_type_count[:i]))
            end = int(sum(cell_type_count[:i+1]))
            for j in range(cell_type_level_constraint.shape[1]):
                expanded_cell_type_level_constraint[i, j] = cell_type_level_constraint[i, j]