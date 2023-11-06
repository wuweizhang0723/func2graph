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
        + str(batch_size)
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
        max_epochs=200,
        gradient_clip_val=0.5,
    )

    trainer.fit(single_model, train_dataloader, val_dataloader)


    # Evaluate the model
    # model_checkpoint_path = checkpoint_path + "/" + listdir(checkpoint_path)[-1]

    # train_results = trainer.predict(single_model, dataloaders=[train_dataloader], ckpt_path=model_checkpoint_path)

    # predictions = []
    # ground_truths = []
    # attentions = []
    # for i in range(len(train_results)):
    #     x_hat = train_results[i][0]    # batch_size * (neuron_num*time)
    #     x = train_results[i][1]
    #     attention = train_results[i][2]

    #     # Need to know the number of batches for each session
    #     # Eache session has a avg attention
    #     attention = attention.view(-1, neuron_num, neuron_num)

    #     predictions.append(x_hat)
    #     ground_truths.append(x)
    #     attentions.append(attention)

    model_checkpoint_path = checkpoint_path + "/" + listdir(checkpoint_path)[-1]
    trained_model = single_model.load_from_checkpoint(model_checkpoint_path)
    trained_model.eval()

    cell_type_level_constraint = trained_model.cell_type_level_constraint.cpu().detach().numpy()

    print(cell_type_level_constraint)

    # fix the order of cell types
    correct_cell_type2id = {'IN':0, 'Vip':1, 'Sncg':2, 'Sst':3, 'EC':4, 'Lamp5':5, 'Serpinf1':6, 'Pvalb':7}
    correct_id2cell_type = {v: k for k, v in correct_cell_type2id.items()}
    cell_type = [correct_id2cell_type.get(i) for i in range(len(correct_cell_type2id))]

    fixed_cell_type_level_constraint = np.zeros((8,8))
    for i in range(8):
        cell_name_i = correct_id2cell_type[i]
        for j in range(8):
            cell_name_j = correct_id2cell_type[j]
            fixed_cell_type_level_constraint[i][j] = cell_type_level_constraint[cell_type2id[cell_name_i]][cell_type2id[cell_name_j]]
            fixed_cell_type_level_constraint[j][i] = cell_type_level_constraint[cell_type2id[cell_name_j]][cell_type2id[cell_name_i]]


    # save the cell type level constraint as npy
    np.save(output_path + "/k_k.npy", fixed_cell_type_level_constraint)


    # Make the ground truth connectivity matrix
    ground_truth_connectivity = np.zeros((8, 8))
    ground_truth_connectivity[:] = np.nan

    ground_truth_connectivity[correct_cell_type2id['EC']][correct_cell_type2id['EC']] = 13/229
    ground_truth_connectivity[correct_cell_type2id['Pvalb']][correct_cell_type2id['EC']] = 22/53
    ground_truth_connectivity[correct_cell_type2id['Sst']][correct_cell_type2id['EC']]= 20/67
    ground_truth_connectivity[correct_cell_type2id['Vip']][correct_cell_type2id['EC']] = 11/68

    ground_truth_connectivity[correct_cell_type2id['EC']][correct_cell_type2id['Pvalb']] = 18/52
    ground_truth_connectivity[correct_cell_type2id['Pvalb']][correct_cell_type2id['Pvalb']] = 45/114
    ground_truth_connectivity[correct_cell_type2id['Sst']][correct_cell_type2id['Pvalb']] = 8/88
    ground_truth_connectivity[correct_cell_type2id['Vip']][correct_cell_type2id['Pvalb']] = 0/54

    ground_truth_connectivity[correct_cell_type2id['EC']][correct_cell_type2id['Sst']] = 13/56
    ground_truth_connectivity[correct_cell_type2id['Pvalb']][correct_cell_type2id['Sst']] = 15/84
    ground_truth_connectivity[correct_cell_type2id['Sst']][correct_cell_type2id['Sst']] = 8/154
    ground_truth_connectivity[correct_cell_type2id['Vip']][correct_cell_type2id['Sst']] = 25/84

    ground_truth_connectivity[correct_cell_type2id['EC']][correct_cell_type2id['Vip']] = 3/62
    ground_truth_connectivity[correct_cell_type2id['Pvalb']][correct_cell_type2id['Vip']] = 1/54
    ground_truth_connectivity[correct_cell_type2id['Sst']][correct_cell_type2id['Vip']] = 12/87
    ground_truth_connectivity[correct_cell_type2id['Vip']][correct_cell_type2id['Vip']] = 2/209

    # get correlation
    fixed_cell_type_level_constraint_ = fixed_cell_type_level_constraint[~np.isnan(ground_truth_connectivity)]
    ground_truth_connectivity_ = ground_truth_connectivity[~np.isnan(ground_truth_connectivity)]
    corr = stats.pearsonr(fixed_cell_type_level_constraint_.flatten(), ground_truth_connectivity_.flatten())[0]
    abs_corr = stats.pearsonr(np.abs(fixed_cell_type_level_constraint_.flatten()), np.abs(ground_truth_connectivity_.flatten()))[0]

    # plot
    plt.imshow(fixed_cell_type_level_constraint, interpolation="nearest")
    plt.colorbar()
    plt.xlabel("Pre")
    plt.ylabel("Post")
    plt.title("Cell type level constraint, corr = " + str(corr))
    plt.xticks(np.arange(len(cell_type)), cell_type, rotation=45)
    plt.yticks(np.arange(len(cell_type)), cell_type)
    plt.savefig(output_path + "/cell_type_level_constraint.png")
    plt.close()

    # plot abs
    plt.imshow(np.abs(fixed_cell_type_level_constraint), interpolation="nearest")
    plt.colorbar()
    plt.xlabel("Pre")
    plt.ylabel("Post")
    plt.title("Cell type level constraint (abs), corr = " + str(abs_corr))
    plt.xticks(np.arange(len(cell_type)), cell_type, rotation=45)
    plt.yticks(np.arange(len(cell_type)), cell_type)
    plt.savefig(output_path + "/cell_type_level_constraint_abs.png")
    plt.close()

    # plot ground truth
    plt.imshow(ground_truth_connectivity, interpolation="nearest")
    plt.colorbar()
    plt.xlabel("Pre")
    plt.ylabel("Post")
    plt.title("Ground truth connectivity")
    plt.xticks(np.arange(len(cell_type)), cell_type, rotation=45)
    plt.yticks(np.arange(len(cell_type)), cell_type)
    plt.savefig(output_path + "/ground_truth_connectivity.png")
    plt.close()