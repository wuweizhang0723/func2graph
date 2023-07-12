import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
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
    parser.add_argument("--tau", help="tau", default=0.025)
    parser.add_argument("--total_time", help="total time", default=100)
    parser.add_argument("--total_data_size", help="total data size", default=1000)

    parser.add_argument("--random_seed", help="random seed", default=42)

    parser.add_argument("--batch_size", help="the batch size", type=int, default=32)

    # Model
    parser.add_argument("--hidden_size_1", help="hidden size 1", default=256)
    parser.add_argument("--h_layers_1", help="h layers 1", default=2)

    parser.add_argument("--heads", help="heads", default=1)
    parser.add_argument("--attention_layers", help="attention layers", default=1)

    parser.add_argument("--hidden_size_2", help="hidden size 2", default=128)
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
    total_time = int(args.total_time)
    total_data_size = int(args.total_data_size)

    random_seed = int(args.random_seed)
    batch_size = int(args.batch_size)