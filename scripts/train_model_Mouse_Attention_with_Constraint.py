import numpy as np
import pandas as pd
from scipy import stats
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
from os import listdir

from func2graph import data, models, baselines, tools


if __name__ == "__main__":
    # Parse arguments add all hyperparameters to the parser
    parser = argparse.ArgumentParser(description="Model Hyperparameters")

    # "Attention_Autoencoder" or "Baseline_2" or "Attention_With_Constraint"
    parser.add_argument(
        "--model_type", type=str, default="Attention_With_Constraint", help="Model type"
    )
    parser.add_argument("--out_folder", help="the output folder")

    # Data

    parser.add_argument("--window_size", default=200)
    parser.add_argument("--predict_window_size", default=1)

    parser.add_argument("--batch_size", help="the batch size", type=int, default=32)

    parser.add_argument("--normalization", default="session")   # "none" or "session" or "neuron" or "log"

    # Model

    parser.add_argument("--model_random_seed", default=42)

    parser.add_argument("--hidden_size_1", help="hidden size 1", default=128)
    parser.add_argument("--h_layers_1", help="h layers 1", default=2)

    parser.add_argument("--heads", help="heads", default=1)
    parser.add_argument("--attention_layers", help="attention layers", default=1)
    parser.add_argument("--dim_key", default=64)

    parser.add_argument("--hidden_size_2", help="hidden size 2", default=258)
    parser.add_argument("--h_layers_2", help="h layers 2", default=2)

    parser.add_argument("--dropout", help="dropout", default=0.2)

    parser.add_argument("--learning_rate", help="learning rate", default=1e-4)

    parser.add_argument("--loss_function", default="mse")   # "mse" or "poisson" or "gaussian"

    parser.add_argument("--constraint_loss_weight", default=1)



    args = parser.parse_args()

    # Set the hyperparameters
    model_type = args.model_type
    out_folder = args.out_folder

    # Data

    window_size = int(args.window_size)
    predict_window_size = int(args.predict_window_size)

    batch_size = int(args.batch_size)

    normalization = args.normalization
    if normalization == "log":
        log_input = True
    else:
        log_input = False

    # Model

    model_random_seed = int(args.model_random_seed)

    hidden_size_1 = int(args.hidden_size_1)
    h_layers_1 = int(args.h_layers_1)

    heads = int(args.heads)
    attention_layers = int(args.attention_layers)
    dim_key = int(args.dim_key)

    hidden_size_2 = int(args.hidden_size_2)
    h_layers_2 = int(args.h_layers_2)

    dropout = float(args.dropout)

    learning_rate = float(args.learning_rate)

    loss_function = args.loss_function

    constraint_loss_weight = float(args.constraint_loss_weight)


    output_path = (
        out_folder
        + model_type
        + "_"
        + str(window_size)
        + "_"
        + str(predict_window_size)
        + "_"
        + normalization
        + "_"
        + str(model_random_seed)
        + "_"
        + str(hidden_size_1)
        + "_"
        + str(h_layers_1)
        + "_"
        + str(heads)
        + "_"
        + str(attention_layers)
        + "_"
        + str(dim_key)
        + "_"
        + str(hidden_size_2)
        + "_"
        + str(h_layers_2)
        + "_"
        + str(dropout)
        + "_"
        + str(learning_rate)
        + "_"
        + loss_function
        + "_"
        + str(constraint_loss_weight)
    )

    checkpoint_path = output_path
    log_path = out_folder + "/log"

    
    train_dataloader, val_dataloader, num_unqiue_neurons, cell_type2id = data.generate_mouse_all_sessions_data(
        window_size=window_size,
        batch_size=batch_size,
        normalization=normalization,
    )

    single_model = models.Attention_With_Constraint(
        num_unqiue_neurons=num_unqiue_neurons,
        num_cell_types=len(cell_type2id),
        model_random_seed=model_random_seed,
        window_size=window_size,
        predict_window_size=predict_window_size,
        hidden_size_1=hidden_size_1,
        h_layers_1=h_layers_1,
        heads=heads,
        attention_layers=attention_layers,
        dim_key=dim_key,
        hidden_size_2=hidden_size_2,
        h_layers_2=h_layers_2,
        dropout=dropout,
        learning_rate=learning_rate,
        loss_function=loss_function,
        log_input=log_input,
        constraint_loss_weight=constraint_loss_weight,
    )

    print(single_model.cell_type_level_constraint.cpu().detach().numpy())


    es = EarlyStopping(monitor="VAL_sum_loss", patience=10)  ###########
    checkpoint_callback = ModelCheckpoint(
        checkpoint_path, monitor="VAL_sum_loss", mode="min", save_top_k=1
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
        max_epochs=400,
    )

    trainer.fit(single_model, train_dataloader, val_dataloader)

    cell_type_level_constraint = single_model.cell_type_level_constraint.cpu().detach().numpy()

    # plot
    plt.imshow(cell_type_level_constraint, interpolation="nearest")
    plt.colorbar()
    plt.xlabel("Pre" + str(cell_type2id))
    plt.ylabel("Post")
    plt.title("Cell type level constraint")
    # plt.xticks(np.arange(len(cell_type2id)), cell_type2id.keys(), rotation=45)
    # plt.yticks(np.arange(len(cell_type2id)), cell_type2id.keys())
    plt.savefig(output_path + "/cell_type_level_constraint.png")
    plt.close()