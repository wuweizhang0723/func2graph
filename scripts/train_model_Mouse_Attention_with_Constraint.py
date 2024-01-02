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

    parser.add_argument("--attention_activation", default="softmax")    # "softmax" or "sigmoid" or "tanh" or "none"

    parser.add_argument("--scheduler", default="plateau")    # "none" or "plateau"

    parser.add_argument("--weight_decay", default=0)

    parser.add_argument("--constraint_var", default=0.04)



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

    attention_activation = args.attention_activation

    scheduler = args.scheduler

    weight_decay = float(args.weight_decay)

    constraint_var = float(args.constraint_var)


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
        + "_"
        + attention_activation
        + "_"
        + scheduler
        + "_"
        + str(weight_decay)
        + "_"
        + str(constraint_var)
    )

    checkpoint_path = output_path
    log_path = out_folder + "/log"

    
    train_dataloader, val_dataloader, num_unqiue_neurons, cell_type2id, num_batch_per_session_TRAIN, num_batch_per_session_VAL, sessions_2_original_cell_type = data.generate_mouse_all_sessions_data(
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
        attention_activation=attention_activation,
        scheduler=scheduler,
        weight_decay=weight_decay,
        constraint_var=constraint_var,
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


    # Evaluate the model --------------------------------------------------------------------------------

    ### n*n => k*k

    model_checkpoint_path = checkpoint_path + "/" + listdir(checkpoint_path)[-1]

    train_results = trainer.predict(single_model, dataloaders=[train_dataloader], ckpt_path=model_checkpoint_path)

    predictions = []
    ground_truths = []
    attentions = []

    all_sessions_avg_attention = []

    index = 0
    for i in range(len(num_batch_per_session_TRAIN)):
        predictions.append([])
        ground_truths.append([])
        attentions.append([])

        for j in range(num_batch_per_session_TRAIN[i]):
            x_hat = train_results[index][0]
            x = train_results[index][1]
            attention = train_results[index][2]

            predictions[i].append(x_hat)
            ground_truths[i].append(x)
            attentions[i].append(attention)

            index += 1

        predictions[i] = torch.cat(predictions[i], dim=0).cpu().numpy()  # N * neuron_num * window_size
        ground_truths[i] = torch.cat(ground_truths[i], dim=0).cpu().numpy()  # N * neuron_num * window_size
        attentions[i] = torch.cat(attentions[i], dim=0).cpu().numpy()    # N * neuron_num * neuron_num

        # get average attention across
        all_sessions_avg_attention.append(np.mean(attentions[i], axis=0))   # neuron_num * neuron_num


    print('hhhh: ', sessions_2_original_cell_type[0])
    print(len(sessions_2_original_cell_type[0]))


    # For each session's avg attention, convert from n*n to k*k attention
    # different sessions may contain slightly different cell types, so k would be different
    all_sessions_k_k_avg_attention = []
    all_sessions_cell_type2cell_type_index = []
    for i in range(len(all_sessions_avg_attention)):
        connectivity_matrix_new, cell_type_id2cell_type, cell_type2cell_type_index, cell_type_count = tools.group_connectivity_matrix_by_cell_type(all_sessions_avg_attention[i], sessions_2_original_cell_type[i])
        connectivity_matrix_cell_type_level = tools.calculate_cell_type_level_connectivity_matrix(connectivity_matrix_new, cell_type_id2cell_type, cell_type_count)
        all_sessions_k_k_avg_attention.append(connectivity_matrix_cell_type_level)
        all_sessions_cell_type2cell_type_index.append(cell_type2cell_type_index)

    corrected_k_k_non_zero_count = np.zeros((8,8))
    corrected_all_sessions_k_k_avg_attention = []    ##### This is what we want !!!!!!!!!!!!
    correct_cell_type2id = {'EC': 0, 'IN': 1, 'Lamp5': 2, 'Pvalb': 3, 'Serpinf1': 4, 'Sncg': 5, 'Sst': 6, 'Vip': 7}
    correct_id2cell_type = {v: k for k, v in correct_cell_type2id.items()}

    print(all_sessions_cell_type2cell_type_index)

    for i in range(len(all_sessions_k_k_avg_attention)):
        corrected_k_k = np.zeros((8,8))
        for j in range(8):
            cell_name_j = correct_id2cell_type[j]
            if cell_name_j not in all_sessions_cell_type2cell_type_index[i]:
                continue
            for k in range(8):
                cell_name_k = correct_id2cell_type[k]
                if cell_name_k not in all_sessions_cell_type2cell_type_index[i]:
                    continue
                corrected_k_k[j][k] = all_sessions_k_k_avg_attention[i][all_sessions_cell_type2cell_type_index[i][cell_name_j]][all_sessions_cell_type2cell_type_index[i][cell_name_k]]
                corrected_k_k[k][j] = all_sessions_k_k_avg_attention[i][all_sessions_cell_type2cell_type_index[i][cell_name_k]][all_sessions_cell_type2cell_type_index[i][cell_name_j]]
                corrected_k_k_non_zero_count[j][k] += 1
                corrected_k_k_non_zero_count[k][j] += 1

        corrected_all_sessions_k_k_avg_attention.append(corrected_k_k)

    FINAL_k_k_avg_attention = np.array(corrected_all_sessions_k_k_avg_attention).sum(axis=0)   # 8 * 8
    print(corrected_k_k_non_zero_count)
    FINAL_k_k_avg_attention = FINAL_k_k_avg_attention / corrected_k_k_non_zero_count

    
    ### k*k prior

    trained_model = single_model.load_from_checkpoint(model_checkpoint_path)
    trained_model.eval()

    cell_type_level_constraint = trained_model.cell_type_level_constraint.cpu().detach().numpy()

    print(cell_type_level_constraint)

    # fix the order of cell types
    cell_type = [correct_id2cell_type.get(i) for i in range(len(correct_cell_type2id))]

    # save the cell type level constraint as npy
    np.save(output_path + "/k_k.npy", cell_type_level_constraint)


    # Make the ground truth connectivity matrix ----------------------------------------------------------
    ground_truth_connectivity = np.zeros((8, 8))
    ground_truth_connectivity[:] = np.nan

    # ground_truth_connectivity[correct_cell_type2id['EC']][correct_cell_type2id['EC']] = 0.3
    # ground_truth_connectivity[correct_cell_type2id['Pvalb']][correct_cell_type2id['EC']] = 0.59
    # ground_truth_connectivity[correct_cell_type2id['Sst']][correct_cell_type2id['EC']]= 0.88
    # ground_truth_connectivity[correct_cell_type2id['Vip']][correct_cell_type2id['EC']] = 1.89

    # ground_truth_connectivity[correct_cell_type2id['EC']][correct_cell_type2id['Pvalb']] = -0.43
    # ground_truth_connectivity[correct_cell_type2id['Pvalb']][correct_cell_type2id['Pvalb']] = -0.53
    # ground_truth_connectivity[correct_cell_type2id['Sst']][correct_cell_type2id['Pvalb']] = -0.60
    # ground_truth_connectivity[correct_cell_type2id['Vip']][correct_cell_type2id['Pvalb']] = -0.44

    # ground_truth_connectivity[correct_cell_type2id['EC']][correct_cell_type2id['Sst']] = -0.31
    # ground_truth_connectivity[correct_cell_type2id['Pvalb']][correct_cell_type2id['Sst']] = -0.43
    # ground_truth_connectivity[correct_cell_type2id['Sst']][correct_cell_type2id['Sst']] = -0.43
    # ground_truth_connectivity[correct_cell_type2id['Vip']][correct_cell_type2id['Sst']] = -0.79

    # ground_truth_connectivity[correct_cell_type2id['EC']][correct_cell_type2id['Vip']] = -0.25
    # ground_truth_connectivity[correct_cell_type2id['Pvalb']][correct_cell_type2id['Vip']] = -0.30
    # ground_truth_connectivity[correct_cell_type2id['Sst']][correct_cell_type2id['Vip']] = -0.42
    # ground_truth_connectivity[correct_cell_type2id['Vip']][correct_cell_type2id['Vip']] = -0.33

    ground_truth_connectivity[correct_cell_type2id['EC']][correct_cell_type2id['EC']] = 0.11
    ground_truth_connectivity[correct_cell_type2id['Pvalb']][correct_cell_type2id['EC']] = 0.27
    ground_truth_connectivity[correct_cell_type2id['Sst']][correct_cell_type2id['EC']]= 0.1
    ground_truth_connectivity[correct_cell_type2id['Vip']][correct_cell_type2id['EC']] = 0.45

    ground_truth_connectivity[correct_cell_type2id['EC']][correct_cell_type2id['Pvalb']] = -0.44
    ground_truth_connectivity[correct_cell_type2id['Pvalb']][correct_cell_type2id['Pvalb']] = -0.47
    ground_truth_connectivity[correct_cell_type2id['Sst']][correct_cell_type2id['Pvalb']] = -0.44
    ground_truth_connectivity[correct_cell_type2id['Vip']][correct_cell_type2id['Pvalb']] = -0.23

    ground_truth_connectivity[correct_cell_type2id['EC']][correct_cell_type2id['Sst']] = -0.16
    ground_truth_connectivity[correct_cell_type2id['Pvalb']][correct_cell_type2id['Sst']] = -0.18
    ground_truth_connectivity[correct_cell_type2id['Sst']][correct_cell_type2id['Sst']] = -0.19
    ground_truth_connectivity[correct_cell_type2id['Vip']][correct_cell_type2id['Sst']] = -0.17

    ground_truth_connectivity[correct_cell_type2id['EC']][correct_cell_type2id['Vip']] = -0.06
    ground_truth_connectivity[correct_cell_type2id['Pvalb']][correct_cell_type2id['Vip']] = -0.10
    ground_truth_connectivity[correct_cell_type2id['Sst']][correct_cell_type2id['Vip']] = -0.17
    ground_truth_connectivity[correct_cell_type2id['Vip']][correct_cell_type2id['Vip']] = -0.10

    # get correlation
    FINAL_k_k_avg_attention_ = FINAL_k_k_avg_attention[~np.isnan(ground_truth_connectivity)]
    ground_truth_connectivity_ = ground_truth_connectivity[~np.isnan(ground_truth_connectivity)]
    corr_avg_k_k = stats.pearsonr(FINAL_k_k_avg_attention_.flatten(), ground_truth_connectivity_.flatten())[0]
    spearman_corr_avg_k_k = stats.spearmanr(FINAL_k_k_avg_attention_.flatten(), ground_truth_connectivity_.flatten())[0]
    abs_corr_avg_k_k = stats.pearsonr(np.abs(FINAL_k_k_avg_attention_.flatten()), np.abs(ground_truth_connectivity_.flatten()))[0]

    fixed_cell_type_level_constraint_ = cell_type_level_constraint[~np.isnan(ground_truth_connectivity)]
    corr = stats.pearsonr(fixed_cell_type_level_constraint_.flatten(), ground_truth_connectivity_.flatten())[0]
    abs_corr = stats.pearsonr(np.abs(fixed_cell_type_level_constraint_.flatten()), np.abs(ground_truth_connectivity_.flatten()))[0]

    # plot
    plt.imshow(FINAL_k_k_avg_attention, interpolation="nearest")
    plt.colorbar()
    plt.xlabel("Pre")
    plt.ylabel("Post")
    plt.title("Avg k by k attention, corr = " + str(corr_avg_k_k)[:7] + ", spearman = " + str(spearman_corr_avg_k_k)[:7])
    plt.xticks(np.arange(len(cell_type)), cell_type, rotation=45)
    plt.yticks(np.arange(len(cell_type)), cell_type)
    plt.savefig(output_path + "/avg_k_k.png")
    plt.close()

    np.save(output_path + "/avg_k_k.npy", FINAL_k_k_avg_attention)

    # plot abs
    plt.imshow(np.abs(FINAL_k_k_avg_attention), interpolation="nearest")
    plt.colorbar()
    plt.xlabel("Pre")
    plt.ylabel("Post")
    plt.title("Avg k by k attention (abs), corr = " + str(abs_corr_avg_k_k))
    plt.xticks(np.arange(len(cell_type)), cell_type, rotation=45)
    plt.yticks(np.arange(len(cell_type)), cell_type)
    plt.savefig(output_path + "/avg_k_k_abs.png")
    plt.close()

    plt.imshow(cell_type_level_constraint, interpolation="nearest")
    plt.colorbar()
    plt.xlabel("Pre")
    plt.ylabel("Post")
    plt.title("Cell type level constraint, corr = " + str(corr))
    plt.xticks(np.arange(len(cell_type)), cell_type, rotation=45)
    plt.yticks(np.arange(len(cell_type)), cell_type)
    plt.savefig(output_path + "/cell_type_level_constraint.png")
    plt.close()

    # plot abs
    plt.imshow(np.abs(cell_type_level_constraint), interpolation="nearest")
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