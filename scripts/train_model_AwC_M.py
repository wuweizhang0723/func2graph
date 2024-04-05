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

    parser.add_argument("--attention_activation", default="softmax")    # "softmax" or "sigmoid" or "tanh" or "none"

    parser.add_argument("--scheduler", default="plateau")    # "none" or "plateau"

    parser.add_argument("--weight_decay", default=0)

    parser.add_argument("--to_q_layers", default=0)
    parser.add_argument("--to_k_layers", default=0)

    parser.add_argument("--constraint_loss_weight", default=1)
    parser.add_argument("--constraint_var", default=0.04)

    parser.add_argument("--causal_temporal_map", default='none')   # 'none', 'off_diagonal_1', 'off_diagonal', 'lower_triangle'
    parser.add_argument("--causal_temporal_map_diff", default=1)   # 1 or 2 or 3, ...
    parser.add_argument("--l1_on_causal_temporal_map", default=0)   # alpha penalty


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

    attention_activation = args.attention_activation

    scheduler = args.scheduler

    weight_decay = float(args.weight_decay)

    to_q_layers = int(args.to_q_layers)
    to_k_layers = int(args.to_k_layers)

    constraint_loss_weight = float(args.constraint_loss_weight)
    constraint_var = float(args.constraint_var)

    causal_temporal_map = args.causal_temporal_map
    causal_temporal_map_diff = int(args.causal_temporal_map_diff)
    l1_on_causal_temporal_map = float(args.l1_on_causal_temporal_map)


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
        + attention_activation
        + "_"
        + scheduler
        + "_"
        + str(weight_decay)
        + "_"
        + str(to_q_layers)
        + "_"
        + str(to_k_layers)
        + "_"
        + str(constraint_loss_weight)
        + "_"
        + str(constraint_var)
        + "_"
        + causal_temporal_map
        + "_"
        + str(causal_temporal_map_diff)
        + "_"
        + str(l1_on_causal_temporal_map)
    )

    checkpoint_path = output_path
    log_path = out_folder + "/log"

    
    train_dataloader, val_dataloader, num_unqiue_neurons, cell_type_order, all_sessions_new_cell_type_id, num_batch_per_session_TRAIN, num_batch_per_session_VAL, sessions_2_original_cell_type, neuron_id_2_cell_type_id = data.generate_mouse_all_sessions_data(
        window_size=window_size,
        batch_size=batch_size,
        normalization=normalization,
    )

    single_model = models.Attention_With_Constraint(
        num_unqiue_neurons=num_unqiue_neurons,
        num_cell_types=len(cell_type_order),
        model_random_seed=model_random_seed,
        window_size=window_size,
        predict_window_size=predict_window_size,
        hidden_size_1=hidden_size_1,
        h_layers_1=h_layers_1,
        heads=heads,
        attention_layers=attention_layers,
        dim_key=dim_key,
        to_q_layers=to_q_layers,
        to_k_layers=to_k_layers,
        hidden_size_2=hidden_size_2,
        h_layers_2=h_layers_2,
        dropout=dropout,
        learning_rate=learning_rate,
        loss_function=loss_function,
        log_input=log_input,
        attention_activation=attention_activation,
        scheduler=scheduler,
        weight_decay=weight_decay,
        constraint_loss_weight=constraint_loss_weight,
        constraint_var=constraint_var,
        causal_temporal_map=causal_temporal_map,
        causal_temporal_map_diff=causal_temporal_map_diff,
        l1_on_causal_temporal_map=l1_on_causal_temporal_map,
    )


    es = EarlyStopping(monitor="VAL_sum_loss", patience=20)  ###########
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


    ############################################################################################################
    # Evaluate the model --------------------------------------------------------------------------------
    ############################################################################################################

    model_checkpoint_path = checkpoint_path + "/" + listdir(checkpoint_path)[-1]

    train_results = trainer.predict(single_model, dataloaders=[train_dataloader], ckpt_path=model_checkpoint_path)

    # neuron_embeddings = train_results[0][3]
    predictions = []
    ground_truths = []
    attentions = []

    all_sessions_avg_attention_NN = []

    index = 0
    num_session = len(num_batch_per_session_TRAIN)
    for i in range(num_session):
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

        # get average attention across samples in each session
        all_sessions_avg_attention_NN.append(np.mean(attentions[i], axis=0))   # neuron_num * neuron_num


    print('hhhh: ', sessions_2_original_cell_type[0])
    print(len(sessions_2_original_cell_type[0]))


    # Perform min-max normalization on each abs(NN) in multisession_NN_list

    multisession_NN_list = all_sessions_avg_attention_NN

    multisession_NN_list_prob = []
    for NN in multisession_NN_list:
        NN = np.abs(NN)
        NN = (NN - np.min(NN)) / (np.max(NN) - np.min(NN))
        multisession_NN_list_prob.append(NN)  


    ############################################################################################################
    ### multisession N*N => ONE K*K
    ###
    ### Use functions in tools.py
    ############################################################################################################

    experiment_KK_prob = tools.multisession_NN_to_KK_1(
        multisession_NN_list_prob, 
        None,
        cell_type_order,
        all_sessions_new_cell_type_id,
    )
    experiment_KK_strength = tools.multisession_NN_to_KK_1(
        multisession_NN_list, 
        None,
        cell_type_order,
        all_sessions_new_cell_type_id,
    )

    eval_cell_type_order = ['EC', 'Pvalb', 'Sst', 'Vip']
    # 1. inferred KK connectivity prob
    eval_KK_prob = tools.experiment_KK_to_eval_KK(experiment_KK_prob, cell_type_order, eval_cell_type_order)
    # 2. inferred KK connectivity strength
    eval_KK_strength = tools.experiment_KK_to_eval_KK(experiment_KK_strength, cell_type_order, eval_cell_type_order)
    

    # For each session's avg attention, convert from n*n to k*k attention
    # different sessions may contain slightly different cell types, so k would be different
    # all_sessions_k_k_avg_attention = []
    # all_sessions_cell_type2cell_type_index = []
    # for i in range(len(all_sessions_avg_attention)):
    #     connectivity_matrix_new, cell_type_id2cell_type, cell_type2cell_type_index, cell_type_count = tools.group_connectivity_matrix_by_cell_type(all_sessions_avg_attention[i], sessions_2_original_cell_type[i])
    #     connectivity_matrix_cell_type_level = tools.calculate_cell_type_level_connectivity_matrix(connectivity_matrix_new, cell_type_id2cell_type, cell_type_count)
    #     all_sessions_k_k_avg_attention.append(connectivity_matrix_cell_type_level)
    #     all_sessions_cell_type2cell_type_index.append(cell_type2cell_type_index)

    # corrected_k_k_non_zero_count = np.zeros((8,8))
    # corrected_all_sessions_k_k_avg_attention = []    ##### This is what we want !!!!!!!!!!!!
    # correct_cell_type2id = {'EC': 0, 'IN': 1, 'Lamp5': 2, 'Pvalb': 3, 'Serpinf1': 4, 'Sncg': 5, 'Sst': 6, 'Vip': 7}
    # correct_id2cell_type = {v: k for k, v in correct_cell_type2id.items()}

    # print(all_sessions_cell_type2cell_type_index)

    # for i in range(len(all_sessions_k_k_avg_attention)):
    #     corrected_k_k = np.zeros((8,8))
    #     for j in range(8):
    #         cell_name_j = correct_id2cell_type[j]
    #         if cell_name_j not in all_sessions_cell_type2cell_type_index[i]:
    #             continue
    #         for k in range(8):
    #             cell_name_k = correct_id2cell_type[k]
    #             if cell_name_k not in all_sessions_cell_type2cell_type_index[i]:
    #                 continue
    #             corrected_k_k[j][k] = all_sessions_k_k_avg_attention[i][all_sessions_cell_type2cell_type_index[i][cell_name_j]][all_sessions_cell_type2cell_type_index[i][cell_name_k]]
    #             corrected_k_k[k][j] = all_sessions_k_k_avg_attention[i][all_sessions_cell_type2cell_type_index[i][cell_name_k]][all_sessions_cell_type2cell_type_index[i][cell_name_j]]
    #             corrected_k_k_non_zero_count[j][k] += 1
    #             corrected_k_k_non_zero_count[k][j] += 1

    #     corrected_all_sessions_k_k_avg_attention.append(corrected_k_k)

    # FINAL_k_k_avg_attention = np.array(corrected_all_sessions_k_k_avg_attention).sum(axis=0)   # 8 * 8
    # print(corrected_k_k_non_zero_count)
    # FINAL_k_k_avg_attention = FINAL_k_k_avg_attention / corrected_k_k_non_zero_count

    
    ### k*k prior

    trained_model = single_model.load_from_checkpoint(model_checkpoint_path)
    trained_model.eval()

    experiment_prior_KK_strength = trained_model.cell_type_level_constraint.cpu().detach().numpy()
    experiment_prior_KK_prob = np.abs(experiment_prior_KK_strength)
    experiment_prior_KK_prob = (experiment_prior_KK_prob - np.min(experiment_prior_KK_prob)) / (np.max(experiment_prior_KK_prob) - np.min(experiment_prior_KK_prob))

    # 3. inferred prior KK connectivity prob
    eval_prior_KK_prob = tools.experiment_KK_to_eval_KK(experiment_prior_KK_prob, cell_type_order, eval_cell_type_order)
    # 4. inferred prior KK connectivity strength
    eval_prior_KK_strength = tools.experiment_KK_to_eval_KK(experiment_prior_KK_strength, cell_type_order, eval_cell_type_order)

    # save the cell type level constraint as npy
    # np.save(output_path + "/k_k.npy", cell_type_level_constraint)

    # fix the order of cell types
    # cell_type = [correct_id2cell_type.get(i) for i in range(len(correct_cell_type2id))]


    # Make the ground truth connectivity matrix ----------------------------------------------------------
    GT_strength_connectivity = np.zeros((len(eval_cell_type_order), len(eval_cell_type_order)))
    GT_strength_connectivity[:] = np.nan

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

    GT_strength_connectivity[eval_cell_type_order.index('EC')][eval_cell_type_order.index('EC')] = 0.11
    GT_strength_connectivity[eval_cell_type_order.index('Pvalb')][eval_cell_type_order.index('EC')] = 0.27
    GT_strength_connectivity[eval_cell_type_order.index('Sst')][eval_cell_type_order.index('EC')]= 0.1
    GT_strength_connectivity[eval_cell_type_order.index('Vip')][eval_cell_type_order.index('EC')] = 0.45

    GT_strength_connectivity[eval_cell_type_order.index('EC')][eval_cell_type_order.index('Pvalb')] = -0.44
    GT_strength_connectivity[eval_cell_type_order.index('Pvalb')][eval_cell_type_order.index('Pvalb')] = -0.47
    GT_strength_connectivity[eval_cell_type_order.index('Sst')][eval_cell_type_order.index('Pvalb')] = -0.44
    GT_strength_connectivity[eval_cell_type_order.index('Vip')][eval_cell_type_order.index('Pvalb')] = -0.23

    GT_strength_connectivity[eval_cell_type_order.index('EC')][eval_cell_type_order.index('Sst')] = -0.16
    GT_strength_connectivity[eval_cell_type_order.index('Pvalb')][eval_cell_type_order.index('Sst')] = -0.18
    GT_strength_connectivity[eval_cell_type_order.index('Sst')][eval_cell_type_order.index('Sst')] = -0.19
    GT_strength_connectivity[eval_cell_type_order.index('Vip')][eval_cell_type_order.index('Sst')] = -0.17

    GT_strength_connectivity[eval_cell_type_order.index('EC')][eval_cell_type_order.index('Vip')] = -0.06
    GT_strength_connectivity[eval_cell_type_order.index('Pvalb')][eval_cell_type_order.index('Vip')] = -0.10
    GT_strength_connectivity[eval_cell_type_order.index('Sst')][eval_cell_type_order.index('Vip')] = -0.17
    GT_strength_connectivity[eval_cell_type_order.index('Vip')][eval_cell_type_order.index('Vip')] = -0.10

    GT_prob_connectivity = np.zeros((len(eval_cell_type_order), len(eval_cell_type_order)))
    GT_prob_connectivity[:] = np.nan

    # replace ground truth prob connectivity with GT prob connectivity
    GT_prob_connectivity[eval_cell_type_order.index('EC')][eval_cell_type_order.index('EC')] = 13/229
    GT_prob_connectivity[eval_cell_type_order.index('Pvalb')][eval_cell_type_order.index('EC')] = 22/53
    GT_prob_connectivity[eval_cell_type_order.index('Sst')][eval_cell_type_order.index('EC')]= 20/67
    GT_prob_connectivity[eval_cell_type_order.index('Vip')][eval_cell_type_order.index('EC')] = 11/68
    
    GT_prob_connectivity[eval_cell_type_order.index('EC')][eval_cell_type_order.index('Pvalb')] = 18/52
    GT_prob_connectivity[eval_cell_type_order.index('Pvalb')][eval_cell_type_order.index('Pvalb')] = 45/114
    GT_prob_connectivity[eval_cell_type_order.index('Sst')][eval_cell_type_order.index('Pvalb')] = 8/88
    GT_prob_connectivity[eval_cell_type_order.index('Vip')][eval_cell_type_order.index('Pvalb')] = 0/54

    GT_prob_connectivity[eval_cell_type_order.index('EC')][eval_cell_type_order.index('Sst')] = 13/56
    GT_prob_connectivity[eval_cell_type_order.index('Pvalb')][eval_cell_type_order.index('Sst')] = 15/84
    GT_prob_connectivity[eval_cell_type_order.index('Sst')][eval_cell_type_order.index('Sst')] = 8/154
    GT_prob_connectivity[eval_cell_type_order.index('Vip')][eval_cell_type_order.index('Sst')] = 25/84

    GT_prob_connectivity[eval_cell_type_order.index('EC')][eval_cell_type_order.index('Vip')] = 3/62
    GT_prob_connectivity[eval_cell_type_order.index('Pvalb')][eval_cell_type_order.index('Vip')] = 1/54
    GT_prob_connectivity[eval_cell_type_order.index('Sst')][eval_cell_type_order.index('Vip')] = 12/87
    GT_prob_connectivity[eval_cell_type_order.index('Vip')][eval_cell_type_order.index('Vip')] = 2/209

    # get correlation
    corr_strength_KK = stats.pearsonr(GT_strength_connectivity.flatten(), eval_KK_strength.flatten())[0]
    spearman_corr_strength_KK = stats.spearmanr(GT_strength_connectivity.flatten(), eval_KK_strength.flatten())[0]

    corr_prob_KK = stats.pearsonr(GT_prob_connectivity.flatten(), eval_KK_prob.flatten())[0]
    spearman_corr_prob_KK = stats.spearmanr(GT_prob_connectivity.flatten(), eval_KK_prob.flatten())[0]

    corr_strength_prior_KK = stats.pearsonr(GT_strength_connectivity.flatten(), eval_prior_KK_strength.flatten())[0]
    spearman_corr_strength_prior_KK = stats.spearmanr(GT_strength_connectivity.flatten(), eval_prior_KK_strength.flatten())[0]

    corr_prob_prior_KK = stats.pearsonr(GT_prob_connectivity.flatten(), eval_prior_KK_prob.flatten())[0]
    spearman_corr_prob_prior_KK = stats.spearmanr(GT_prob_connectivity.flatten(), eval_prior_KK_prob.flatten())[0]

    # plot
    plt.imshow(eval_KK_strength, interpolation="nearest")
    plt.colorbar()
    plt.xlabel("Pre")
    plt.ylabel("Post")
    plt.title("eval_KK_strength, corr = " + str(corr_strength_KK)[:7] + ", spearman = " + str(spearman_corr_strength_KK)[:7])
    plt.xticks(np.arange(len(eval_cell_type_order)), eval_cell_type_order, rotation=45)
    plt.yticks(np.arange(len(eval_cell_type_order)), eval_cell_type_order)
    plt.savefig(output_path + "/strength.png")
    plt.close()

    np.save(output_path + "/Estimated_strength.npy", eval_KK_strength)

    # plot
    plt.imshow(eval_KK_prob, interpolation="nearest", cmap='bone')
    plt.colorbar()
    plt.xlabel("Pre")
    plt.ylabel("Post")
    plt.title("eval_KK_prob, corr = " + str(corr_prob_KK)[:7] + ", spearman = " + str(spearman_corr_prob_KK)[:7])
    plt.xticks(np.arange(len(eval_cell_type_order)), eval_cell_type_order, rotation=45)
    plt.yticks(np.arange(len(eval_cell_type_order)), eval_cell_type_order)
    plt.savefig(output_path + "/prob.png")
    plt.close()

    np.save(output_path + "/Estimated_prob.npy", eval_KK_prob)

    # plot
    plt.imshow(eval_prior_KK_strength, interpolation="nearest")
    plt.colorbar()
    plt.xlabel("Pre")
    plt.ylabel("Post")
    plt.title("eval_prior_KK_strength, corr = " + str(corr_strength_prior_KK)[:7] + ", spearman = " + str(spearman_corr_strength_prior_KK)[:7])
    plt.xticks(np.arange(len(eval_cell_type_order)), eval_cell_type_order, rotation=45)
    plt.yticks(np.arange(len(eval_cell_type_order)), eval_cell_type_order)
    plt.savefig(output_path + "/prior_strength.png")
    plt.close()

    np.save(output_path + "/Estimated_prior_strength.npy", eval_prior_KK_strength)

    # plot
    plt.imshow(eval_prior_KK_prob, interpolation="nearest", cmap='bone')
    plt.colorbar()
    plt.xlabel("Pre")
    plt.ylabel("Post")
    plt.title("eval_prior_KK_prob, corr = " + str(corr_prob_prior_KK)[:7] + ", spearman = " + str(spearman_corr_prob_prior_KK)[:7])
    plt.xticks(np.arange(len(eval_cell_type_order)), eval_cell_type_order, rotation=45)
    plt.yticks(np.arange(len(eval_cell_type_order)), eval_cell_type_order)
    plt.savefig(output_path + "/prior_prob.png")
    plt.close()

    # plot ground truth
    plt.imshow(GT_strength_connectivity, interpolation="nearest")
    plt.colorbar()
    plt.xlabel("Pre")
    plt.ylabel("Post")
    plt.title("Ground truth strength connectivity")
    plt.xticks(np.arange(len(eval_cell_type_order)), eval_cell_type_order, rotation=45)
    plt.yticks(np.arange(len(eval_cell_type_order)), eval_cell_type_order)
    plt.savefig(output_path + "/GT_strength.png")
    plt.close()

    plt.imshow(GT_prob_connectivity, interpolation="nearest", cmap='bone')
    plt.colorbar()
    plt.xlabel("Pre")
    plt.ylabel("Post")
    plt.title("Ground truth prob connectivity")
    plt.xticks(np.arange(len(eval_cell_type_order)), eval_cell_type_order, rotation=45)
    plt.yticks(np.arange(len(eval_cell_type_order)), eval_cell_type_order)
    plt.savefig(output_path + "/GT_prob.png")