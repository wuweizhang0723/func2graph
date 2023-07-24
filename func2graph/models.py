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



class Base(pl.LightningModule):
    def __init__(self, scheduler="plateau") -> None:
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
                    patience=3,
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
        x = batch[0]   # batch_size * (neuron_num*time)
        x_hat = self(x)
        loss = F.mse_loss(x_hat, x, reduction="mean")

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[0]   # batch_size * (neuron_num*time)
        x_hat = self(x)
        loss = F.mse_loss(x_hat, x, reduction="mean")

        self.log("val_loss", loss)
        result = torch.stack([x_hat.cpu().detach(), x.cpu().detach()], dim=1)
        return result
    
    def test_step(self, batch, batch_idx):
        x = batch[0]   # batch_size * (neuron_num*time)
        x_hat = self(x)
        loss = F.mse_loss(x_hat, x)
        self.log("test_loss", loss)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x = batch[0]   # batch_size * (neuron_num*time)
        x_hat, attention = self(x)
        return x_hat, x, attention
    





class Attention_Autoencoder(Base):
    def __init__(
        self,
        neuron_num=10,
        window_size=200,
        hidden_size_1=128, # MLP_1
        h_layers_1=2,
        heads=1,  # Attention
        attention_layers=1,
        hidden_size_2=256, # MLP_2
        h_layers_2=2,
        dropout=0.2,
        learning_rate=1e-4,
        prediction_mode=False,
        pos_enc_type="none",  # "sin_cos" or "lookup_table" or "none"
    ):
        super().__init__()
        self.save_hyperparameters()

        # MLP_1

        self.fc1 = nn.Sequential(
            nn.Linear(window_size, hidden_size_1), nn.ReLU()
        )

        self.fclayers1 = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden_size_1, hidden_size_1), nn.ReLU(), nn.Dropout(dropout)
            )
            for layer in range(h_layers_1)
        )

        # Attention

        self.pos_enc_type = pos_enc_type
        if pos_enc_type == "sin_cos":
            self.position_enc = PositionalEncoding(hidden_size_1, neuron_num=neuron_num)
            self.layer_norm = nn.LayerNorm(hidden_size_1)
        elif pos_enc_type == "lookup_table":
            self.embedding_table = nn.Embedding(
                num_embeddings=neuron_num, embedding_dim=hidden_size_1
            )
            self.layer_norm = nn.LayerNorm(hidden_size_1)

        self.attentionlayers = nn.ModuleList()

        for layer in range(attention_layers):
            self.attentionlayers.append(
                nn.Sequential(
                    Residual_For_Attention(
                        Attention(
                            dim=hidden_size_1,  # dimension of the last out channel
                            heads=heads,
                            prediction_mode=prediction_mode,
                        ),
                        prediction_mode=prediction_mode,
                    ),
                )
            )
            self.attentionlayers.append(
                nn.Sequential(
                    nn.LayerNorm(hidden_size_1),
                    Residual(
                        nn.Sequential(
                            nn.Linear(hidden_size_1, hidden_size_1 * 2),
                            nn.Dropout(dropout),
                            nn.ReLU(),
                            nn.Linear(hidden_size_1 * 2, hidden_size_1),
                            nn.Dropout(dropout),
                        )
                    ),
                    nn.LayerNorm(hidden_size_1),
                )
            )

        # MLP_2

        self.fc2 = nn.Sequential(
            nn.Linear(hidden_size_1, hidden_size_2), nn.ReLU()
        )

        self.fclayers2 = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden_size_2, hidden_size_2), nn.ReLU(), nn.Dropout(dropout)
            )
            for layer in range(h_layers_2)
        )

        self.out = nn.Linear(hidden_size_2, window_size)


    def forward(self, x): # x: batch_size * (neuron_num*time)
        x = self.fc1(x)
        for layer in self.fclayers1:
            x = layer(x)

        if self.pos_enc_type == "sin_cos":
            # Add positional encoding
            x = self.position_enc(x)
            x = self.layer_norm(x)
        elif self.pos_enc_type == "lookup_table":
            # Add positional encoding
            idx = torch.arange(x.shape[1]).to(x.device)
            x = x + self.embedding_table(idx)
            x = self.layer_norm(x)

        attention_results = []
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
        