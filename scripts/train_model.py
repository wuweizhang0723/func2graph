import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning import loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from func2graph import data
from func2graph import models



if __name__ == "__main__":
    # Parse arguments add all hyperparameters to the parser
    parser = argparse.ArgumentParser(description="Model Hyperparameters")
    parser.add_argument(
        "--model_type", type=str, default="Attention_Autoencoder", help="Model type"
    )
    parser.add_argument("--out_folder", help="the output folder")

    # Data
    parser.add_argument("--neuron_num", help="the number of neurons", type=int, default=10)
    parser.add_argument("--dt", help="dt", default=0.001)
    parser.add_argument("--tau", help="tau", default=0.3)
    parser.add_argument("--spike_neuron_num", default=2)
    parser.add_argument("--spike_input", default=5)

    parser.add_argument("--weight_scale", default=5)
    parser.add_argument("--init_scale", default=0.1)

    parser.add_argument("--total_time", help="total time", default=30000)
    parser.add_argument("--random_seed", help="random seed", default=42)

    parser.add_argument("--train_data_size", default=20000)
    parser.add_argument("--window_size", default=200)

    parser.add_argument("--batch_size", help="the batch size", type=int, default=32)

    parser.add_argument("--data_type", default="reconstruction")
    parser.add_argument("--predict_window_size", default=100)

    # Model
    parser.add_argument("--hidden_size_1", help="hidden size 1", default=128)
    parser.add_argument("--h_layers_1", help="h layers 1", default=2)

    parser.add_argument("--heads", help="heads", default=1)
    parser.add_argument("--attention_layers", help="attention layers", default=1)

    parser.add_argument("--hidden_size_2", help="hidden size 2", default=258)
    parser.add_argument("--h_layers_2", help="h layers 2", default=2)

    parser.add_argument("--dropout", help="dropout", default=0.2)

    parser.add_argument("--learning_rate", help="learning rate", default=1e-4)


    args = parser.parse_args()

    # Set the hyperparameters
    model_type = args.model_type
    out_folder = args.out_folder

    # Data
    neuron_num = int(args.neuron_num)
    dt = float(args.dt)
    tau = float(args.tau)
    spike_neuron_num = int(args.spike_neuron_num)
    spike_input = int(args.spike_input)

    weight_scale = float(args.weight_scale)
    init_scale = float(args.init_scale)

    total_time = int(args.total_time)
    random_seed = int(args.random_seed)

    train_data_size = int(args.train_data_size)
    window_size = int(args.window_size)

    batch_size = int(args.batch_size)

    data_type = args.data_type
    predict_window_size = int(args.predict_window_size)

    # Model
    hidden_size_1 = int(args.hidden_size_1)
    h_layers_1 = int(args.h_layers_1)

    heads = int(args.heads)
    attention_layers = int(args.attention_layers)

    hidden_size_2 = int(args.hidden_size_2)
    h_layers_2 = int(args.h_layers_2)

    dropout = float(args.dropout)

    learning_rate = float(args.learning_rate)



    output_path = (
        out_folder
        + model_type
        + "_"
        + str(neuron_num)
        + "_"
        + str(dt)
        + "_"
        + str(tau)
        + "_"
        + str(spike_neuron_num)
        + "_"
        + str(spike_input)
        + "_"
        + str(weight_scale)
        + "_"
        + str(init_scale)
        + "_"
        + str(total_time)
        + "_"
        + str(random_seed)
        + "_"
        + str(train_data_size)
        + "_"
        + str(window_size)
        + "_"
        + data_type
        + "_"
        + str(predict_window_size)
        + "_"
        + str(hidden_size_1)
        + "_"
        + str(h_layers_1)
        + "_"
        + str(heads)
        + "_"
        + str(attention_layers)
        + "_"
        + str(hidden_size_2)
        + "_"
        + str(h_layers_2)
        + "_"
        + str(learning_rate)
    )

    checkpoint_path = output_path
    log_path = out_folder + "/log"

    trainloader, validloader, weight_matrix = data.generate_simulation_data(
        neuron_num=neuron_num,
        dt=dt,
        tau=tau,
        weight_scale=weight_scale,
        init_scale=init_scale,
        total_time=total_time,
        spike_neuron_num=spike_neuron_num,
        spike_input=spike_input,
        train_data_size=train_data_size,
        window_size=window_size,
        random_seed=random_seed,
        batch_size=batch_size,
        data_type=data_type,
        predict_window_size=predict_window_size,
    )

    if model_type == "Attention_Autoencoder":
        single_model = models.Attention_Autoencoder(
            neuron_num=neuron_num,
            window_size=window_size,
            hidden_size_1=hidden_size_1,
            h_layers_1=h_layers_1,
            heads=heads,
            attention_layers=attention_layers,
            hidden_size_2=hidden_size_2,
            h_layers_2=h_layers_2,
            dropout=dropout,
            learning_rate=learning_rate,
            data_type=data_type,
            predict_window_size=predict_window_size,
        )

    es = EarlyStopping(monitor="val_loss", patience=15)
    checkpoint_callback = ModelCheckpoint(
        checkpoint_path, monitor="val_loss", mode="min", save_top_k=1
    )
    lr_monitor = LearningRateMonitor()
    logger = TensorBoardLogger(log_path, name="model")
    trainer = pl.Trainer(
        devices=[0],
        accelerator="gpu",
        callbacks=[es, checkpoint_callback, lr_monitor],
        benchmark=False,
        profiler="simple",
        logger=logger,
    )

    trainer.fit(single_model, trainloader, validloader)