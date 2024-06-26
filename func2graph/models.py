import numpy as np
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from func2graph.layers import (
    Causal_Temporal_Map_Attention,
    Causal_Temporal_Map_Attention_2,
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
        x, y = batch
        y_hat = self(x)
        
        # pred = y_hat[:, :, -1:]
        pred = y_hat
        target = y

        if self.hparams.loss_function == "mse":
            loss = F.mse_loss(pred, target, reduction="mean") + self.hparams.l1_on_causal_temporal_map * sum([p.abs().sum() for p in self.attentionlayers[0][0].W_Q_W_KT.parameters()])
        elif self.hparams.loss_function == "poisson":
            loss = F.poisson_nll_loss(pred, target, log_input=self.hparams.log_input, reduction="mean")
        elif self.hparams.loss_function == "gaussian":
            var = torch.ones(pred.shape, requires_grad=True).to(pred.device)  ##############################
            loss = F.gaussian_nll_loss(pred, target, reduction="mean", var=var)

        self.log(str(self.hparams.loss_function) + " train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
        # pred = y_hat[:, :, -1:]
        pred = y_hat
        target = y

        result = torch.stack([pred.cpu().detach(), target.cpu().detach()], dim=1)

        if self.hparams.loss_function == "mse":
            loss = F.mse_loss(pred, target, reduction="mean")
        elif self.hparams.loss_function == "poisson":
            loss = F.poisson_nll_loss(pred, target, log_input=self.hparams.log_input, reduction="mean")
        elif self.hparams.loss_function == "gaussian":
            var = torch.ones(pred.shape, requires_grad=True).to(pred.device)  ##############################
            loss = F.gaussian_nll_loss(pred, target, reduction="mean", var=var)

        self.log(str(self.hparams.loss_function) + " val_loss", loss)
        return result
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
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
        x, y = batch
        y_hat, attention, neuron_embedding = self(x)

        return y_hat, y, attention, neuron_embedding
    





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
        hidden_size_2=64, # MLP_2
        h_layers_2=0,
        dropout=0.2,
        learning_rate=1e-4,
        scheduler="plateau",
        prediction_mode=False,
        pos_enc_type="lookup_table",  # "lookup_table" or "none"
        predict_window_size = 100,
        loss_function = "mse", # "mse" or "poisson" or "gaussian"
        log_input = False,
        attention_activation = "none", # "softmax" or "sigmoid" or "tanh", "none"
        weight_decay = 0,
        causal_temporal_map = 'none',  # 'none', 'off_diagonal_1', 'off_diagonal', 'lower_triangle'
        causal_temporal_map_diff = 1,
        l1_on_causal_temporal_map = 0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.prediction_mode = prediction_mode

        self.predict_window_size = predict_window_size

        torch.manual_seed(model_random_seed)

        # MLP_1

        hidden_size_1 = window_size - predict_window_size


        # hidden_size_1 = window_size - predict_window_size       #######################

        # self.fc1 = nn.Sequential(
        #     nn.Linear(window_size - predict_window_size, hidden_size_1), nn.ReLU()
        # )

        # self.fclayers1 = nn.ModuleList(
        #     nn.Sequential(
        #         nn.Linear(hidden_size_1, hidden_size_1), nn.ReLU(), nn.Dropout(dropout)
        #     )
        #     for layer in range(h_layers_1)
        # )


        # Attention

        self.pos_enc_type = pos_enc_type
        if pos_enc_type == "lookup_table":
            self.embedding_table = nn.Embedding(
                num_embeddings=neuron_num, embedding_dim=hidden_size_1
            )
            dim_in = hidden_size_1
            self.layer_norm = nn.LayerNorm(dim_in)
        else:
            dim_in = hidden_size_1

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
            #if causal_temporal_map != 'none':
            self.attentionlayers.append(
                nn.Sequential(
                    Causal_Temporal_Map_Attention(
                        dim=dim_in,  # the last dimension of input
                        prediction_mode=self.prediction_mode,
                        activation = attention_activation,
                        causal_temporal_map = causal_temporal_map,
                        diff = causal_temporal_map_diff,
                    ),
                )
            )
            # else:
            #     self.attentionlayers.append(
            #         nn.Sequential(
            #             Attention(
            #                 dim=dim_in,  # the last dimension of input
            #                 to_q_layers=to_q_layers,
            #                 to_k_layers=to_k_layers,
            #                 heads=heads,
            #                 dim_key=dim_key,
            #                 prediction_mode=self.prediction_mode,
            #                 activation = attention_activation,
            #             ),
            #         )
            #     )

            # Layer norm after attention layer makes training loss more stable
            self.attentionlayers.append(
                nn.Sequential(
                    nn.LayerNorm(dim_in),
                    # Residual(
                    #     nn.Sequential(
                    #         nn.Linear(dim_in, hidden_size_1 * 2),
                    #         nn.Dropout(dropout),
                    #         nn.ReLU(),
                    #         nn.Linear(hidden_size_1 * 2, dim_in),
                    #         nn.Dropout(dropout),
                    #         nn.ReLU(),
                    #     )
                    # ),
                    # nn.LayerNorm(dim_in),
                )
            )

        # MLP_2

        # self.fc2 = nn.Sequential(
        #     nn.Linear(dim_in, hidden_size_2), nn.ReLU()
        # )

        # self.fclayers2 = nn.ModuleList(
        #     nn.Sequential(
        #         nn.Linear(hidden_size_2, hidden_size_2), nn.ReLU(), nn.Dropout(dropout)
        #     )
        #     for layer in range(h_layers_2)
        # )

        # self.out = nn.Linear(hidden_size_2, predict_window_size)


    def forward(self, x): # x: batch_size * (neuron_num*time)
        # x = self.fc1(x)
        # for layer in self.fclayers1:
        #     x = layer(x)

        if self.pos_enc_type == "lookup_table":
            # Add positional encoding
            idx = torch.arange(x.shape[1]).to(x.device)
            neuron_embedding = self.embedding_table(idx)
            x = x + neuron_embedding
            # embedding = einops.repeat(self.embedding_table(idx), 'n d -> b n d', b=x.shape[0])
            # x = torch.concat([x, embedding], dim=-1)
            
            x = self.layer_norm(x)   ########################

        attention_results = []
        print('length: ', len(self.attentionlayers))
        for layer in self.attentionlayers:
            x = layer(x)
            if type(x) is tuple:
                print('3')
                x, attn = x
                attention_results.append(attn)

        # x = self.fc2(x)
        # for layer in self.fclayers2:
        #     x = layer(x)
                

        if self.hparams.prediction_mode == True:
            # return x[:, :, -1*self.predict_window_size:], attention_results[0], neuron_embedding.clone()
            return x[:, :, -1*self.predict_window_size:], attention_results[0], neuron_embedding.clone()
        else:
            return x[:, :, -1*self.predict_window_size:]




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
        cell_type_count = [int(self.hparams.neuron_num*0.76), int(self.hparams.neuron_num*0.08), int(self.hparams.neuron_num*0.08), int(self.hparams.neuron_num*0.08)]

        x, y = batch
        y_hat, neuron_level_attention, cell_type_level_constraint = self(x)
        
        pred = y_hat
        target = y

        expanded_cell_type_level_constraint = torch.zeros((neuron_level_attention.shape[1],neuron_level_attention.shape[2]), requires_grad=True).to(pred.device)
        accumulated_count_row = 0
        for i in range(len(cell_type_count)):
            accumulated_count_row += cell_type_count[i]
            accumulated_count_col = 0
            for j in range(len(cell_type_count)):
                accumulated_count_col += cell_type_count[j]
                expanded_cell_type_level_constraint[accumulated_count_row-cell_type_count[i]:accumulated_count_row, accumulated_count_col-cell_type_count[j]:accumulated_count_col] = cell_type_level_constraint[i, j]

        # Expand the first dimension of expanded_cell_type_level_constraint to batch_size
        expanded_cell_type_level_constraint = einops.repeat(expanded_cell_type_level_constraint, 'n d -> b n d', b=neuron_level_attention.shape[0])

        # Use Gaussian NLL Loss to add constraint, var should be a hyperparameter
        var_constraint = torch.ones(neuron_level_attention.shape, requires_grad=True).to(pred.device) * self.hparams.constraint_var
        constraint_loss = F.gaussian_nll_loss(neuron_level_attention, expanded_cell_type_level_constraint, reduction="mean", var=var_constraint)

        if self.hparams.loss_function == "mse":
            loss = F.mse_loss(pred, target, reduction="mean") + self.hparams.l1_on_causal_temporal_map * sum([p.abs().sum() for p in self.attentionlayers[0][0].W_Q_W_KT.parameters()])
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
        cell_type_count = [int(self.hparams.neuron_num*0.76), int(self.hparams.neuron_num*0.08), int(self.hparams.neuron_num*0.08), int(self.hparams.neuron_num*0.08)]

        x, y = batch
        y_hat, neuron_level_attention, cell_type_level_constraint = self(x)
        
        pred = y_hat
        target = y

        expanded_cell_type_level_constraint = torch.zeros((neuron_level_attention.shape[1],neuron_level_attention.shape[2]), requires_grad=True).to(pred.device)
        accumulated_count_row = 0
        for i in range(len(cell_type_count)):
            accumulated_count_row += cell_type_count[i]
            accumulated_count_col = 0
            for j in range(len(cell_type_count)):
                accumulated_count_col += cell_type_count[j]
                expanded_cell_type_level_constraint[(accumulated_count_row-cell_type_count[i]):accumulated_count_row, (accumulated_count_col-cell_type_count[j]):accumulated_count_col] = cell_type_level_constraint[i, j]

        # Expand the first dimension of expanded_cell_type_level_constraint to batch_size
        expanded_cell_type_level_constraint = einops.repeat(expanded_cell_type_level_constraint, 'n d -> b n d', b=neuron_level_attention.shape[0])

        # Use Gaussian NLL Loss to add constraint, var should be a hyperparameter
        var_constraint = torch.ones(neuron_level_attention.shape, requires_grad=True).to(pred.device) * self.hparams.constraint_var
        constraint_loss = F.gaussian_nll_loss(neuron_level_attention, expanded_cell_type_level_constraint, reduction="mean", var=var_constraint)

        result = torch.stack([pred.cpu().detach(), target.cpu().detach()], dim=1)

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
        return result

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y_hat, neuron_level_attention, cell_type_level_constraint = self(x)

        return y_hat, y, neuron_level_attention, cell_type_level_constraint



class Attention_With_Constraint_sim(Base_3):
    def __init__(
        self,
        model_random_seed=42,
        neuron_num=200,
        num_cell_types=4,
        window_size=200,
        learning_rate=1e-4,
        scheduler="plateau",
        pos_enc_type="lookup_table",  # "lookup_table" or "none"
        predict_window_size=1,
        loss_function = "mse", # "mse" or "poisson" or "gaussian"
        attention_activation = "none", # "softmax" or "sigmoid" or "tanh", "none"
        weight_decay = 0,
        causal_temporal_map = 'none',  # 'none', 'off_diagonal_1', 'off_diagonal', 'lower_triangle'
        causal_temporal_map_diff = 1,
        l1_on_causal_temporal_map = 0,
        constraint_loss_weight = 0,
        constraint_var = 1,
        out_layer=False,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.predict_window_size = predict_window_size
        self.neuron_num = neuron_num

        torch.manual_seed(model_random_seed)

        # k * k matrix constraint
        self.cell_type_level_constraint = nn.Parameter(torch.FloatTensor(num_cell_types, num_cell_types).uniform_(-1, 1))

        hidden_size_1 = window_size - predict_window_size

        # Attention

        self.pos_enc_type = pos_enc_type
        if pos_enc_type == "lookup_table":
            self.embedding_table = nn.Embedding(
                num_embeddings=neuron_num, embedding_dim=hidden_size_1
            )
            dim_in = hidden_size_1
            self.layer_norm = nn.LayerNorm(dim_in)
        else:
            dim_in = hidden_size_1

        dim_in = hidden_size_1

        self.attentionlayers = nn.ModuleList()
        self.attentionlayers.append(
            nn.Sequential(
                Causal_Temporal_Map_Attention(
                    dim=dim_in,  # the last dimension of input
                    prediction_mode=True,
                    activation = attention_activation,
                    causal_temporal_map = causal_temporal_map,
                    diff = causal_temporal_map_diff,
                ),
            )
        )
        self.attentionlayers.append(
            nn.Sequential(
                nn.LayerNorm(dim_in),
            )
        )

        self.out_layer = out_layer
        if out_layer == True:
            self.out = nn.Linear(dim_in, predict_window_size, bias=False)

    def forward(self, x): # x: batch_size * (neuron_num*time)

        # Add positional encoding
        idx = torch.arange(x.shape[1]).to(x.device)
        neuron_embedding = self.embedding_table(idx)

        attention_results = []
        x, attn, attn3 = self.attentionlayers[0][0](x, neuron_embedding)
        attention_results.append(attn)

        x = self.attentionlayers[1][0](x)
        
        if self.out_layer == True:
            return self.out(x), attention_results[0], self.cell_type_level_constraint
        else:
            return x[:, :, -1*self.predict_window_size:], attention_results[0], self.cell_type_level_constraint




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
        pred, neuron_level_attention, cell_type_level_constraint, neuron_level_attention3 = self(x[:, :, :-1*self.hparams.predict_window_size], neuron_ids)

        cell_type_ids_np = cell_type_ids[0].clone().detach().cpu().numpy()
        expanded_cell_type_level_constraint = torch.zeros((neuron_level_attention.shape[1],neuron_level_attention.shape[2]), requires_grad=True).to(pred.device)
        expanded_cell_type_level_var = torch.zeros((neuron_level_attention.shape[1],neuron_level_attention.shape[2]), requires_grad=True).to(pred.device)
        # loop over unique cell types

        for i in list(np.unique(cell_type_ids_np)):
            # find the neurons with the same cell type
            neuron_ids_with_same_cell_type_i = np.where(cell_type_ids_np == i)[0]
            for j in list(np.unique(cell_type_ids_np)):
                # find the neurons with the same cell type
                neuron_ids_with_same_cell_type_j = np.where(cell_type_ids_np == j)[0]
                # Assign the same constraint to the neurons with the same cell type
                expanded_cell_type_level_constraint[neuron_ids_with_same_cell_type_i, neuron_ids_with_same_cell_type_j.reshape(-1,1)] = cell_type_level_constraint[i, j] * 1   # to create computational graph
                expanded_cell_type_level_constraint[neuron_ids_with_same_cell_type_j, neuron_ids_with_same_cell_type_i.reshape(-1,1)] = cell_type_level_constraint[j, i] * 1

                expanded_cell_type_level_var[neuron_ids_with_same_cell_type_i, neuron_ids_with_same_cell_type_j.reshape(-1,1)] = self.cell_type_level_var[i, j] ** 2   # to create computational graph
                expanded_cell_type_level_var[neuron_ids_with_same_cell_type_j, neuron_ids_with_same_cell_type_i.reshape(-1,1)] = self.cell_type_level_var[j, i] ** 2

    
        # print('num zeros in expanded_cell_type_level_constraint: ', torch.sum(expanded_cell_type_level_constraint == 0))
    
        # Expand the first dimension of expanded_cell_type_level_constraint to batch_size
        expanded_cell_type_level_constraint = einops.repeat(expanded_cell_type_level_constraint, 'n d -> b n d', b=neuron_level_attention.shape[0])
        expanded_cell_type_level_var = einops.repeat(expanded_cell_type_level_var, 'n d -> b n d', b=neuron_level_attention.shape[0])

        ############################################## Prior
        
        # 0. Default
        # Use Gaussian NLL Loss to add constraint, var should be a hyperparameter
        constraint_loss = F.gaussian_nll_loss(neuron_level_attention, expanded_cell_type_level_constraint, reduction="mean", var=expanded_cell_type_level_var)

        # 1. Constrain attention to be (0,1) or (0, self.dist_var)
        # var_constraint = torch.ones(neuron_level_attention.shape, requires_grad=True).to(pred.device)
        # var_constraint = self.dist_var * var_constraint
        # mean_constraint = torch.zeros(neuron_level_attention.shape, requires_grad=False).to(pred.device)
        # constraint_loss = F.gaussian_nll_loss(neuron_level_attention, mean_constraint, reduction="mean", var=var_constraint)

        # 2. KK prior constrain attention, dist prior constrain attention to be (0,1)
        # constraint_loss_1 = F.gaussian_nll_loss(neuron_level_attention, expanded_cell_type_level_constraint, reduction="mean", var=expanded_cell_type_level_var)
        # var_constraint = torch.ones(neuron_level_attention.shape, requires_grad=False).to(pred.device)
        # mean_constraint = torch.zeros(neuron_level_attention.shape, requires_grad=False).to(pred.device)
        # constraint_loss_2 = F.gaussian_nll_loss(neuron_level_attention, mean_constraint, reduction="mean", var=var_constraint)
        # constraint_loss = constraint_loss_1 + constraint_loss_2

        # 3. kk prior constrain attention, dist prior constrain kk prior to be (0,1)
        # constraint_loss_1 = F.gaussian_nll_loss(neuron_level_attention, expanded_cell_type_level_constraint, reduction="mean", var=expanded_cell_type_level_var)
        # var_constraint = torch.ones(cell_type_level_constraint.shape, requires_grad=False).to(pred.device)
        # mean_constraint = torch.zeros(cell_type_level_constraint.shape, requires_grad=False).to(pred.device)
        # constraint_loss_2 = F.gaussian_nll_loss(cell_type_level_constraint, mean_constraint, reduction="mean", var=var_constraint)
        # constraint_loss = constraint_loss_1 + constraint_loss_2

        # 4. summing over neuron_level_attention, constrain it to be (0,self.dist_var)
        # attn_sum = torch.sum(neuron_level_attention)
        # mean_constraint = torch.zeros(1, requires_grad=False).to(pred.device)
        # constraint_loss = F.mse_loss(attn_sum, mean_constraint, reduction="mean")

        ##############################################
        
        # make pred and target have the same shape
        target = target.reshape(pred.shape)

        if self.hparams.loss_function == "mse":
            loss = F.mse_loss(pred, target, reduction="mean") + self.hparams.l1_on_causal_temporal_map * sum([p.abs().sum() for p in self.attentionlayers[0][0].W_Q_W_KT.parameters()])
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
        pred, neuron_level_attention, cell_type_level_constraint, neuron_level_attention3 = self(x[:, :, :-1*self.hparams.predict_window_size], neuron_ids)

        cell_type_ids_np = cell_type_ids[0].clone().detach().cpu().numpy()
        expanded_cell_type_level_constraint = torch.zeros((neuron_level_attention.shape[1],neuron_level_attention.shape[2]), requires_grad=True).to(pred.device)
        expanded_cell_type_level_var = torch.zeros((neuron_level_attention.shape[1],neuron_level_attention.shape[2]), requires_grad=True).to(pred.device)
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

                expanded_cell_type_level_var[neuron_ids_with_same_cell_type_i, neuron_ids_with_same_cell_type_j.reshape(-1,1)] = self.cell_type_level_var[i, j] ** 2   # to create computational graph
                expanded_cell_type_level_var[neuron_ids_with_same_cell_type_j, neuron_ids_with_same_cell_type_i.reshape(-1,1)] = self.cell_type_level_var[j, i] ** 2

            
        expanded_cell_type_level_constraint = einops.repeat(expanded_cell_type_level_constraint, 'n d -> b n d', b=neuron_level_attention.shape[0])
        expanded_cell_type_level_var = einops.repeat(expanded_cell_type_level_var, 'n d -> b n d', b=neuron_level_attention.shape[0])

        ############################################## Prior

        constraint_loss = F.gaussian_nll_loss(neuron_level_attention, expanded_cell_type_level_constraint, reduction="mean", var=expanded_cell_type_level_var)

        # 1. Constrain attention to be (0,1) or (0, self.dist_var)
        # var_constraint = torch.ones(neuron_level_attention.shape, requires_grad=True).to(pred.device)
        # var_constraint = self.dist_var * var_constraint
        # mean_constraint = torch.zeros(neuron_level_attention.shape, requires_grad=False).to(pred.device)
        # constraint_loss = F.gaussian_nll_loss(neuron_level_attention, mean_constraint, reduction="mean", var=var_constraint)

        # 2. KK prior constrain attention, dist prior constrain attention to be (0,1)
        # constraint_loss_1 = F.gaussian_nll_loss(neuron_level_attention, expanded_cell_type_level_constraint, reduction="mean", var=expanded_cell_type_level_var)
        # var_constraint = torch.ones(neuron_level_attention.shape, requires_grad=False).to(pred.device)
        # mean_constraint = torch.zeros(neuron_level_attention.shape, requires_grad=False).to(pred.device)
        # constraint_loss_2 = F.gaussian_nll_loss(neuron_level_attention, mean_constraint, reduction="mean", var=var_constraint)
        # constraint_loss = constraint_loss_1 + constraint_loss_2

        # 3. kk prior constrain attention, dist prior constrain kk prior to be (0,1)
        # constraint_loss_1 = F.gaussian_nll_loss(neuron_level_attention, expanded_cell_type_level_constraint, reduction="mean", var=expanded_cell_type_level_var)
        # var_constraint = torch.ones(cell_type_level_constraint.shape, requires_grad=False).to(pred.device)
        # mean_constraint = torch.zeros(cell_type_level_constraint.shape, requires_grad=False).to(pred.device)
        # constraint_loss_2 = F.gaussian_nll_loss(cell_type_level_constraint, mean_constraint, reduction="mean", var=var_constraint)
        # constraint_loss = constraint_loss_1 + constraint_loss_2

        # 4. summing over neuron_level_attention, constrain it to be (0,self.dist_var)
        # attn_sum = torch.sum(neuron_level_attention)
        # mean_constraint = torch.zeros(1, requires_grad=True).to(pred.device)
        # constraint_loss = F.mse_loss(attn_sum, mean_constraint, reduction="mean")

        ##############################################
        
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

        pred, neuron_level_attention, cell_type_level_constraint, neuron_level_attention3 = self(x[:, :, :-1*self.hparams.predict_window_size], neuron_ids)
        return pred, target, neuron_level_attention, neuron_level_attention3
    


class Attention_With_Constraint(Base_2):
    def __init__(
        self,
        num_unqiue_neurons,
        num_cell_types,
        model_type="Attention_With_Constraint",
        model_random_seed=42,
        window_size=200,
        h_layers_1=2,
        dim_key=64,   # Attention
        learning_rate=1e-4,
        scheduler="cycle",
        predict_window_size = 1,
        loss_function = "mse", # "mse" or "poisson" or "gaussian"
        log_input = False,
        attention_activation = "none", # "softmax" or "sigmoid" or "tanh"
        weight_decay = 0,
        causal_temporal_map = 'none',  # 'none', 'off_diagonal', 'lower_triangle'
        causal_temporal_map_diff = 1,
        l1_on_causal_temporal_map = 0,
        constraint_loss_weight = 0,
        constraint_var = 1,
    ):
        super().__init__()
        self.save_hyperparameters()

        torch.manual_seed(model_random_seed)

        # k * k matrix constraint
        # self.cell_type_level_constraint = nn.Parameter(torch.zeros((num_cell_types, num_cell_types)), requires_grad=True)
        self.cell_type_level_constraint = nn.Parameter(torch.FloatTensor(num_cell_types, num_cell_types).uniform_(-1, 1))
        # self.cell_type_level_constraint = nn.Parameter(torch.FloatTensor(num_cell_types, num_cell_types).uniform_(0, 1))
        self.cell_type_level_var = nn.Parameter(torch.ones(num_cell_types, num_cell_types), requires_grad=True)

        # single value constraint
        # self.dist_var = nn.Parameter(torch.ones(1), requires_grad=True)

        self.predict_window_size = predict_window_size

        hidden_size_1 = window_size - predict_window_size

        # Attention

        self.embedding_table = nn.Embedding(
            num_embeddings=num_unqiue_neurons, embedding_dim=hidden_size_1   # global unique neuron lookup table
        )
        dim_in = hidden_size_1

        self.attentionlayers = nn.ModuleList()

        self.attentionlayers.append(
            nn.Sequential(
                Causal_Temporal_Map_Attention(
                    dim=dim_in,
                    prediction_mode=True,
                    activation = attention_activation,
                    causal_temporal_map = causal_temporal_map,
                    diff = causal_temporal_map_diff,
                )
            )
        )
        self.attentionlayers.append(
            nn.Sequential(
                nn.LayerNorm(dim_in),
            )
        )

        # self.out = nn.Linear(dim_in, predict_window_size, bias=False)

        # self.out = nn.Parameter(torch.FloatTensor(dim_in, predict_window_size).uniform_(0, 1))
        # self.out_relu = nn.ReLU()
        # self.out_softmax = nn.Softmax(dim=0)

    def forward(self, x, neuron_ids): # x: batch_size * (neuron_num*time), neuron_ids: batch_size * neuron_num
        # Add positional encoding
        e = self.embedding_table(neuron_ids[0])   # the first dimension doesn't matter

        attention_results = []
        attention3_results = []
        output, attn, attn3 = self.attentionlayers[0][0](x, e)
        attention_results.append(attn)
        attention3_results.append(attn3)

        output = self.attentionlayers[1][0](output)

        # output = output + (x+e)       ################### residual connection

        # x = self.out(x)

        batch_neuron_num = attention_results[0].shape[-1]
        return output[:, :, -1*self.predict_window_size:], attention_results[0].view(-1, batch_neuron_num, batch_neuron_num), self.cell_type_level_constraint, attention3_results[0].view(-1, batch_neuron_num, batch_neuron_num)
        # return x @ self.out_relu(self.out), attention_results[0].view(-1, batch_neuron_num, batch_neuron_num), self.cell_type_level_constraint
        # return x @ torch.abs(self.out), attention_results[0].view(-1, batch_neuron_num, batch_neuron_num), self.cell_type_level_constraint
        # return x @ self.out_softmax(self.out), attention_results[0].view(-1, batch_neuron_num, batch_neuron_num), self.cell_type_level_constraint

        # return x @ self.out, attention_results[0].view(-1, batch_neuron_num, batch_neuron_num), self.cell_type_level_constraint

        # self.out.weight.data = self.out_relu(self.out.weight.data)
        # return self.out(x), attention_results[0].view(-1, batch_neuron_num, batch_neuron_num), self.cell_type_level_constraint

        # return x @ self.out_relu(self.out), attention_results[0].view(-1, batch_neuron_num, batch_neuron_num), self.cell_type_level_constraint
    




class Attention_With_Constraint_2(Base_2):
    def __init__(
        self,
        num_unqiue_neurons,
        num_cell_types,
        model_type="Attention_With_Constraint_2",
        model_random_seed=42,
        window_size=200,
        predict_window_size = 1,
        dropout=0.2,
        learning_rate=1e-4,
        scheduler="cycle",
        loss_function = "mse", # "mse" or "poisson" or "gaussian"
        weight_decay = 0,
        causal_temporal_map = 'none',  # 'none', 'off_diagonal', 'lower_triangle'
        causal_temporal_map_diff = 1,
        l1_on_causal_temporal_map = 0,
        constraint_loss_weight = 0,
        constraint_var = 1,
        dim_E = 128,
    ):
        super().__init__()
        self.save_hyperparameters()

        torch.manual_seed(model_random_seed)

        self.cell_type_level_constraint = nn.Parameter(torch.FloatTensor(num_cell_types, num_cell_types).uniform_(-1, 1))
        self.cell_type_level_var = nn.Parameter(torch.ones(num_cell_types, num_cell_types), requires_grad=True)

        self.predict_window_size = predict_window_size
        dim_X = window_size - predict_window_size

        # Attention

        self.embedding_table = nn.Embedding(
            num_embeddings=num_unqiue_neurons, embedding_dim=dim_E   # global unique neuron lookup table
        )

        self.layer_norm = nn.LayerNorm(dim_X)

        self.attentionlayers = nn.ModuleList()
        self.attentionlayers.append(
            nn.Sequential(
                Causal_Temporal_Map_Attention_2(
                    dim_X=dim_X,
                    dim_E=dim_E,
                    prediction_mode=True,
                    causal_temporal_map = causal_temporal_map,
                    diff = causal_temporal_map_diff,
                ),
                nn.LayerNorm(dim_X),
            )
        )

        # self.out = nn.Linear(dim_X+dim_E, predict_window_size, bias=False)
        # self.out_relu = nn.ReLU()

        # self.out = nn.Parameter(torch.FloatTensor(dim_X, predict_window_size).uniform_(0, 1))

    def forward(self, x, neuron_ids): # x: batch_size * (neuron_num*time), neuron_ids: batch_size * neuron_num
        e = self.embedding_table(neuron_ids[0])
        x = self.layer_norm(x)

        attention_results = []
        attention3_results = []
        x, attn, attn3 = self.attentionlayers[0][0](x, e)
        attention_results.append(attn)
        attention3_results.append(attn3)

        x = self.attentionlayers[0][1](x)

        batch_neuron_num = attention_results[0].shape[-1]
        # self.out.weight.data = self.out_relu(self.out.weight.data)
        return x[:, :, -1*self.predict_window_size:], attention_results[0].view(-1, batch_neuron_num, batch_neuron_num), self.cell_type_level_constraint, attention3_results[0].view(-1, batch_neuron_num, batch_neuron_num)
        # return self.out(x), attention_results[0].view(-1, batch_neuron_num, batch_neuron_num), self.cell_type_level_constraint, attention3_results[0].view(-1, batch_neuron_num, batch_neuron_num)