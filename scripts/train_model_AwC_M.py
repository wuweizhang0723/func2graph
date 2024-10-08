import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from os import listdir
from sklearn.metrics import r2_score

from func2graph import data, models, tools


if __name__ == "__main__":
    # Parse arguments add all hyperparameters to the parser
    parser = argparse.ArgumentParser(description="Model Hyperparameters")

    # "Attention_With_Constraint", "Attention_With_Constraint_2"
    parser.add_argument(
        "--model_type", type=str, default="Attention_With_Constraint", help="Model type"
    )
    parser.add_argument("--out_folder", help="the output folder")

    # Data

    parser.add_argument("--input_mouse")
    parser.add_argument("--input_sessions")

    parser.add_argument("--window_size", default=200)
    parser.add_argument("--predict_window_size", default=1)

    parser.add_argument("--batch_size", help="the batch size", type=int, default=32)

    parser.add_argument("--normalization", default="session")   # "none" or "session" or "neuron" or "log"

    # Model

    parser.add_argument("--model_random_seed", default=42)

    parser.add_argument("--h_layers_1", help="h layers 1", default=2)

    parser.add_argument("--dim_key", default=64)

    parser.add_argument("--learning_rate", help="learning rate", default=1e-4)

    parser.add_argument("--loss_function", default="mse")   # "mse" or "poisson" or "gaussian"

    parser.add_argument("--attention_activation", default="softmax")    # "softmax" or "sigmoid" or "tanh" or "none"

    parser.add_argument("--scheduler", default="plateau")    # "none" or "plateau"

    parser.add_argument("--weight_decay", default=0)

    parser.add_argument("--constraint_loss_weight", default=1)
    parser.add_argument("--constraint_var", default=0.04)

    parser.add_argument("--causal_temporal_map", default='none')   # 'none', 'off_diagonal_1', 'off_diagonal', 'lower_triangle'
    parser.add_argument("--causal_temporal_map_diff", default=1)   # 1 or 2 or 3, ...
    parser.add_argument("--l1_on_causal_temporal_map", default=0)   # alpha penalty

    parser.add_argument("--dim_E", default=100)   ##### This is for Attention_With_Constraint_2 model


    args = parser.parse_args()

    # Set the hyperparameters
    model_type = args.model_type
    out_folder = args.out_folder

    # Data

    input_mouse = [str(mouse) for mouse in args.input_mouse.split('|')]
    input_sessions = [str(mouse) for mouse in args.input_sessions.split('|')]
    for i in range(len(input_sessions)):
        input_sessions[i] = input_sessions[i].split('_')
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

    h_layers_1 = int(args.h_layers_1)

    dim_key = int(args.dim_key)

    learning_rate = float(args.learning_rate)
    loss_function = args.loss_function

    attention_activation = args.attention_activation

    scheduler = args.scheduler

    weight_decay = float(args.weight_decay)

    constraint_loss_weight = float(args.constraint_loss_weight)
    constraint_var = float(args.constraint_var)

    causal_temporal_map = args.causal_temporal_map
    causal_temporal_map_diff = int(args.causal_temporal_map_diff)
    l1_on_causal_temporal_map = float(args.l1_on_causal_temporal_map)

    dim_E = int(args.dim_E)


    output_path = (
        out_folder
        + model_type
        + "_"
        + args.input_mouse
        + "_"
        + args.input_sessions
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
        + str(h_layers_1)
        + "_"
        + str(dim_key)
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
        + str(constraint_loss_weight)
        + "_"
        + str(constraint_var)
        + "_"
        + causal_temporal_map
        + "_"
        + str(causal_temporal_map_diff)
        + "_"
        + str(l1_on_causal_temporal_map)
        + "_"
        + str(dim_E)
    )

    checkpoint_path = output_path
    log_path = out_folder + "/log"

    
    train_dataloader, val_dataloader, num_unqiue_neurons, cell_type_order, all_sessions_new_cell_type_id, num_batch_per_session_TRAIN, num_batch_per_session_VAL, sessions_2_original_cell_type, neuron_id_2_cell_type_id = data.generate_mouse_all_sessions_data(
        input_mouse=input_mouse,
        input_sessions=input_sessions,
        window_size=window_size,
        batch_size=batch_size,
    )

    if model_type == "Attention_With_Constraint":
        single_model = models.Attention_With_Constraint(
            num_unqiue_neurons=num_unqiue_neurons,
            num_cell_types=len(cell_type_order),
            model_random_seed=model_random_seed,
            window_size=window_size,
            predict_window_size=predict_window_size,
            h_layers_1=h_layers_1,
            dim_key=dim_key,
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
    elif model_type == "Attention_With_Constraint_2":
        single_model = models.Attention_With_Constraint_2(
            num_unqiue_neurons=num_unqiue_neurons,
            num_cell_types=len(cell_type_order),
            model_random_seed=model_random_seed,
            window_size=window_size,
            predict_window_size=predict_window_size,
            learning_rate=learning_rate,
            scheduler=scheduler,
            loss_function=loss_function,
            attention_activation=attention_activation,
            weight_decay=weight_decay,
            constraint_loss_weight=constraint_loss_weight,
            constraint_var=constraint_var,
            causal_temporal_map=causal_temporal_map,
            causal_temporal_map_diff=causal_temporal_map_diff,
            l1_on_causal_temporal_map=l1_on_causal_temporal_map,
            dim_E=dim_E,
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
        max_epochs=100,  # 100, 400
        gradient_clip_val=0,
    )

    trainer.fit(single_model, train_dataloader, val_dataloader)


    ############################################################################################################
    # Evaluate the model --------------------------------------------------------------------------------
    ############################################################################################################

    eval_cell_type_order = ['EC', 'Pvalb', 'Sst', 'Vip']
    GT_strength_connectivity = np.zeros((len(eval_cell_type_order), len(eval_cell_type_order)))
    GT_strength_connectivity[:] = np.nan

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

    max_abs = np.max(np.abs(GT_strength_connectivity))
    vmin_KK = -max_abs
    vmax_KK = max_abs



    model_checkpoint_path = checkpoint_path + "/" + listdir(checkpoint_path)[-1]

    train_results = trainer.predict(single_model, dataloaders=[train_dataloader], ckpt_path=model_checkpoint_path)

    # neuron_embeddings = train_results[0][3]
    # predictions = []
    # ground_truths = []
    attentions = []  # list of (N * neuron_num * neuron_num)
    # attentions_3 = []
    attentions_by_state = []  # list of (3 * N * neuron_num * neuron_num)

    all_sessions_avg_attention_NN = []  # list of (neuron_num * neuron_num)
    # all_sessions_avg_attention_NN_3 = []
    all_sessions_avg_attention_NN_by_state = []  # list of (3 * neuron_num * neuron_num)

    index = 0
    num_session = len(num_batch_per_session_TRAIN)
    for i in range(num_session):
        # predictions.append([])
        # ground_truths.append([])
        attentions.append([])
        # attentions_3.append([])
        attentions_by_state.append([[], [], []]) # 3 states for each session

        for j in range(num_batch_per_session_TRAIN[i]):
            x_hat = train_results[index][0]
            x = train_results[index][1]
            attention = train_results[index][2]  # B * neuron_num * neuron_num
            state = train_results[index][4].cpu().numpy()   # B * window_size
            
            # predictions[i].append(x_hat)
            # ground_truths[i].append(x)
            attentions[i].append(attention)

            # attention_3 = train_results[index][3]
            # attentions_3[i].append(attention_3)

            # check if all values in state of a sample are the same
            for k in range(state.shape[0]):
                if np.all(state[k] == state[k][0]):
                    attentions_by_state[i][state[k][0]].append(attention[k])
            
            index += 1

        # predictions[i] = torch.cat(predictions[i], dim=0).cpu().numpy()  # N * neuron_num * window_size
        # ground_truths[i] = torch.cat(ground_truths[i], dim=0).cpu().numpy()  # N * neuron_num * window_size
        attentions[i] = torch.cat(attentions[i], dim=0).cpu().numpy()    # N * neuron_num * neuron_num

        print('attentions: ', attentions[i].shape)

        attentions_by_state[i][0] = torch.stack(attentions_by_state[i][0], dim=0).cpu().numpy()    # N * neuron_num * neuron_num
        attentions_by_state[i][1] = torch.stack(attentions_by_state[i][1], dim=0).cpu().numpy()    # N * neuron_num * neuron_num
        attentions_by_state[i][2] = torch.stack(attentions_by_state[i][2], dim=0).cpu().numpy()    # N * neuron_num * neuron_num

        print('attentions_by_state 0: ', attentions_by_state[i][0].shape)
        print('attentions_by_state 1: ', attentions_by_state[i][1].shape)
        print('attentions_by_state 2: ', attentions_by_state[i][2].shape)

        # get average attention across samples in each session
        all_sessions_avg_attention_NN.append(np.mean(attentions[i], axis=0))   # neuron_num * neuron_num

        all_sessions_avg_attention_NN_by_state.append([])
        all_sessions_avg_attention_NN_by_state[i].append(np.mean(attentions_by_state[i][0], axis=0))   # neuron_num * neuron_num
        all_sessions_avg_attention_NN_by_state[i].append(np.mean(attentions_by_state[i][1], axis=0))   # neuron_num * neuron_num
        all_sessions_avg_attention_NN_by_state[i].append(np.mean(attentions_by_state[i][2], axis=0))   # neuron_num * neuron_num

        all_sessions_avg_attention_NN_by_state[i] = np.stack(all_sessions_avg_attention_NN_by_state[i], axis=0)  # 3 * neuron_num * neuron_num

        # attentions_3[i] = torch.cat(attentions_3[i], dim=0).cpu().numpy()    # N * neuron_num * neuron_num
        # all_sessions_avg_attention_NN_3.append(np.mean(attentions_3[i], axis=0))   # neuron_num * neuron_num


    print('hhhh: ', sessions_2_original_cell_type[0])
    print(len(sessions_2_original_cell_type[0]))

    print('attentions_by_state 0: ', attentions_by_state[0][0].shape)
    print('attentions_by_state 1: ', attentions_by_state[0][1].shape)
    print('attentions_by_state 2: ', attentions_by_state[0][2].shape)

    # get all avg attentions from the same state
    state2all_sessions_avg_attention = [[], [], []]   # 3 * num_session * neuron_num * neuron_num
    for i in range(3):
       state2all_sessions_avg_attention[i] = [all_sessions_avg_attention_NN_by_state[j][i] for j in range(num_session)]

    experiment_KK_strength_0 = tools.multisession_NN_to_KK_1(
        state2all_sessions_avg_attention[0],
        None,
        cell_type_order,
        all_sessions_new_cell_type_id,
    )
    experiment_KK_strength_1 = tools.multisession_NN_to_KK_1(
        state2all_sessions_avg_attention[1],
        None,
        cell_type_order,
        all_sessions_new_cell_type_id,
    )
    experiment_KK_strength_2 = tools.multisession_NN_to_KK_1(
        state2all_sessions_avg_attention[2],
        None,
        cell_type_order,
        all_sessions_new_cell_type_id,
    )

    eval_KK_strength_0 = tools.experiment_KK_to_eval_KK(experiment_KK_strength_0, cell_type_order, eval_cell_type_order)
    eval_KK_strength_1 = tools.experiment_KK_to_eval_KK(experiment_KK_strength_1, cell_type_order, eval_cell_type_order)
    eval_KK_strength_2 = tools.experiment_KK_to_eval_KK(experiment_KK_strength_2, cell_type_order, eval_cell_type_order)

    vmin_across_states = min(np.min(eval_KK_strength_0), np.min(eval_KK_strength_1), np.min(eval_KK_strength_2))
    vmax_across_states = max(np.max(eval_KK_strength_0), np.max(eval_KK_strength_1), np.max(eval_KK_strength_2))

    plt.imshow(eval_KK_strength_0, cmap='RdBu_r', interpolation="nearest", vmin=vmin_across_states, vmax=vmax_across_states)
    # plt.colorbar()
    plt.xlabel("Pre")
    plt.ylabel("Post")
    plt.title("eval_KK_strength_0")
    plt.xticks(np.arange(len(eval_cell_type_order)), eval_cell_type_order, rotation=45)
    plt.yticks(np.arange(len(eval_cell_type_order)), eval_cell_type_order)
    plt.savefig(output_path + "/strength_0.png")
    plt.close()

    plt.imshow(eval_KK_strength_1, cmap='RdBu_r', interpolation="nearest", vmin=vmin_across_states, vmax=vmax_across_states)
    # plt.colorbar()
    plt.xlabel("Pre")
    plt.ylabel("Post")
    plt.title("eval_KK_strength_1")
    plt.xticks(np.arange(len(eval_cell_type_order)), eval_cell_type_order, rotation=45)
    plt.yticks(np.arange(len(eval_cell_type_order)), eval_cell_type_order)
    plt.savefig(output_path + "/strength_1.png")
    plt.close()

    plt.imshow(eval_KK_strength_2, cmap='RdBu_r', interpolation="nearest", vmin=vmin_across_states, vmax=vmax_across_states)
    # plt.colorbar()
    plt.xlabel("Pre")
    plt.ylabel("Post")
    plt.title("eval_KK_strength_2")
    plt.xticks(np.arange(len(eval_cell_type_order)), eval_cell_type_order, rotation=45)
    plt.yticks(np.arange(len(eval_cell_type_order)), eval_cell_type_order)
    plt.savefig(output_path + "/strength_2.png")
    plt.close()




    ################################### validation result

    val_results = trainer.predict(single_model, dataloaders=[val_dataloader], ckpt_path=model_checkpoint_path)

    predictions = []
    ground_truths = []

    index = 0
    num_session = len(num_batch_per_session_VAL)
    for i in range(num_session):
        predictions.append([])
        ground_truths.append([])

        for j in range(num_batch_per_session_VAL[i]):
            x_hat = val_results[index][0]
            x = val_results[index][1]
            
            predictions[i].append(x_hat)
            ground_truths[i].append(x)
            index += 1
        
        predictions[i] = torch.cat(predictions[i], dim=0).cpu().numpy()  # N * neuron_num * window_size
        ground_truths[i] = torch.cat(ground_truths[i], dim=0).cpu().numpy()  # N * neuron_num * window_size

    flatten_predictions = [predictions[0].flatten()]
    flatten_ground_truths = [ground_truths[0].flatten()]
    for i in range(1, num_session):
        flatten_predictions.append(predictions[i].flatten())
        flatten_ground_truths.append(ground_truths[i].flatten())

    flatten_predictions = np.concatenate(flatten_predictions)
    flatten_ground_truths = np.concatenate(flatten_ground_truths)
    pred_corr = stats.pearsonr(flatten_predictions, flatten_ground_truths)[0]

    R_squared = r2_score(flatten_ground_truths, flatten_predictions)

    mse = np.mean((flatten_predictions - flatten_ground_truths) ** 2)

    # plot the prediction and groundtruth curve for 5 neurons
    for i in range(10):
        plt.subplot(10, 1, i+1)
        print("hhhh " + str(predictions[0].shape))
        plt.plot(predictions[0][:200, i, 0], label="Prediction")
        plt.plot(ground_truths[0][:200, i, 0], label="Ground Truth")
    plt.legend()
    plt.savefig(output_path + "/curve.png")
    plt.close()


    plt.scatter(flatten_predictions, flatten_ground_truths, s=1)
    plt.title("val_corr = " + str(pred_corr)[:7] + ", R^2 = " + str(R_squared)[:7] + ", mse = " + str(mse)[:7])
    plt.xlabel("Predictions")
    plt.ylabel("Ground Truths")
    plt.savefig(output_path + "/pred.png")
    plt.close()


    ################################### Perform min-max normalization on each abs(NN) in multisession_NN_list

    multisession_NN_list = all_sessions_avg_attention_NN

    # multisession_NN_list_prob = []
    # for NN in multisession_NN_list:
    #     # NN = np.abs(NN)
    #     mean = np.mean(NN)
    #     min_val, max_val = np.min(NN), np.max(NN)
    #     range = max(np.abs(max_val - mean), np.abs(min_val - mean))
    #     NN = np.abs(NN - mean) / range
    #     # threshold = 0.5
    #     # NN[NN < threshold] = 0
    #     # NN[NN >= threshold] = 1
    #     multisession_NN_list_prob.append(NN)  


    ############################################################################################################
    ### multisession N*N => ONE K*K
    ###
    ### Use functions in tools.py
    ############################################################################################################

    # experiment_KK_prob = tools.multisession_NN_to_KK_1(
    #     multisession_NN_list_prob, 
    #     None,
    #     cell_type_order,
    #     all_sessions_new_cell_type_id,
    # )
    experiment_KK_strength = tools.multisession_NN_to_KK_1(
        multisession_NN_list, 
        None,
        cell_type_order,
        all_sessions_new_cell_type_id,
    )

    eval_cell_type_order = ['EC', 'Pvalb', 'Sst', 'Vip']
    # 1. inferred KK connectivity prob
    # eval_KK_prob = tools.experiment_KK_to_eval_KK(experiment_KK_prob, cell_type_order, eval_cell_type_order)
    # 2. inferred KK connectivity strength
    eval_KK_strength = tools.experiment_KK_to_eval_KK(experiment_KK_strength, cell_type_order, eval_cell_type_order)
    

    # Attention_3
    # multisession_NN_3_list = all_sessions_avg_attention_NN_3
    # experiment_KK_3_strength = tools.multisession_NN_to_KK_1(
    #     multisession_NN_3_list,
    #     None,
    #     cell_type_order,
    #     all_sessions_new_cell_type_id,
    # )

    # eval_KK_3_strength = tools.experiment_KK_to_eval_KK(experiment_KK_3_strength, cell_type_order, eval_cell_type_order)


    
    ### k*k prior

    trained_model = single_model.load_from_checkpoint(model_checkpoint_path)
    trained_model.eval()

    experiment_prior_KK_strength = trained_model.cell_type_level_constraint.clone().detach().cpu().numpy()
    experiment_prior_KK_prob = np.abs(experiment_prior_KK_strength)
    experiment_prior_KK_prob = (experiment_prior_KK_prob - np.min(experiment_prior_KK_prob)) / (np.max(experiment_prior_KK_prob) - np.min(experiment_prior_KK_prob))

    experiment_prior_KK_var = trained_model.cell_type_level_var.clone().detach().cpu().numpy()
    experiment_prior_KK_var = experiment_prior_KK_var ** 2

    # 3. inferred prior KK connectivity prob
    eval_prior_KK_prob = tools.experiment_KK_to_eval_KK(experiment_prior_KK_prob, cell_type_order, eval_cell_type_order)
    # 4. inferred prior KK connectivity strength
    eval_prior_KK_strength = tools.experiment_KK_to_eval_KK(experiment_prior_KK_strength, cell_type_order, eval_cell_type_order)

    eval_prior_KK_var = tools.experiment_KK_to_eval_KK(experiment_prior_KK_var, cell_type_order, eval_cell_type_order)


    # Make the ground truth connectivity matrix ----------------------------------------------------------
    # GT_prob_connectivity = np.zeros((len(eval_cell_type_order), len(eval_cell_type_order)))
    # GT_prob_connectivity[:] = np.nan

    # replace ground truth prob connectivity with GT prob connectivity
    # GT_prob_connectivity[eval_cell_type_order.index('EC')][eval_cell_type_order.index('EC')] = 13/229
    # GT_prob_connectivity[eval_cell_type_order.index('Pvalb')][eval_cell_type_order.index('EC')] = 22/53
    # GT_prob_connectivity[eval_cell_type_order.index('Sst')][eval_cell_type_order.index('EC')]= 20/67
    # GT_prob_connectivity[eval_cell_type_order.index('Vip')][eval_cell_type_order.index('EC')] = 11/68
    
    # GT_prob_connectivity[eval_cell_type_order.index('EC')][eval_cell_type_order.index('Pvalb')] = 18/52
    # GT_prob_connectivity[eval_cell_type_order.index('Pvalb')][eval_cell_type_order.index('Pvalb')] = 45/114
    # GT_prob_connectivity[eval_cell_type_order.index('Sst')][eval_cell_type_order.index('Pvalb')] = 8/88
    # GT_prob_connectivity[eval_cell_type_order.index('Vip')][eval_cell_type_order.index('Pvalb')] = 0/54

    # GT_prob_connectivity[eval_cell_type_order.index('EC')][eval_cell_type_order.index('Sst')] = 13/56
    # GT_prob_connectivity[eval_cell_type_order.index('Pvalb')][eval_cell_type_order.index('Sst')] = 15/84
    # GT_prob_connectivity[eval_cell_type_order.index('Sst')][eval_cell_type_order.index('Sst')] = 8/154
    # GT_prob_connectivity[eval_cell_type_order.index('Vip')][eval_cell_type_order.index('Sst')] = 25/84

    # GT_prob_connectivity[eval_cell_type_order.index('EC')][eval_cell_type_order.index('Vip')] = 3/62
    # GT_prob_connectivity[eval_cell_type_order.index('Pvalb')][eval_cell_type_order.index('Vip')] = 1/54
    # GT_prob_connectivity[eval_cell_type_order.index('Sst')][eval_cell_type_order.index('Vip')] = 12/87
    # GT_prob_connectivity[eval_cell_type_order.index('Vip')][eval_cell_type_order.index('Vip')] = 2/209

    # get correlation
    corr_strength_KK = stats.pearsonr(GT_strength_connectivity.flatten(), eval_KK_strength.flatten())[0]
    spearman_corr_strength_KK = stats.spearmanr(GT_strength_connectivity.flatten(), eval_KK_strength.flatten())[0]

    # corr_prob_KK = stats.pearsonr(GT_prob_connectivity.flatten(), eval_KK_prob.flatten())[0]
    # spearman_corr_prob_KK = stats.spearmanr(GT_prob_connectivity.flatten(), eval_KK_prob.flatten())[0]

    corr_strength_prior_KK = stats.pearsonr(GT_strength_connectivity.flatten(), eval_prior_KK_strength.flatten())[0]
    spearman_corr_strength_prior_KK = stats.spearmanr(GT_strength_connectivity.flatten(), eval_prior_KK_strength.flatten())[0]

    # corr_prob_prior_KK = stats.pearsonr(GT_prob_connectivity.flatten(), eval_prior_KK_prob.flatten())[0]
    # spearman_corr_prob_prior_KK = stats.spearmanr(GT_prob_connectivity.flatten(), eval_prior_KK_prob.flatten())[0]


    # corr_strength_KK_3 = stats.pearsonr(GT_strength_connectivity.flatten(), eval_KK_3_strength.flatten())[0]
    # spearman_corr_strength_KK_3 = stats.spearmanr(GT_strength_connectivity.flatten(), eval_KK_3_strength.flatten())[0]


    ############################################################# TT matrix evaluation
    if model_type == "Attention_With_Constraint":
        TT = trained_model.attentionlayers[0][0].W_Q_W_KT.weight.cpu().detach().numpy()
        TT = TT.T

        plt.imshow(TT, cmap='bone', interpolation="nearest")
        plt.title("W_Q @ W_K^T")
        plt.colorbar()
        plt.savefig(output_path + "/TT.png")
        plt.close()

        np.save(output_path + "/TT.npy", TT)

    ############################################################ plot
    plt.imshow(eval_prior_KK_var, interpolation="nearest")
    plt.colorbar()
    plt.xlabel("Pre")
    plt.ylabel("Post")
    plt.title("eval_prior_KK_var")
    plt.xticks(np.arange(len(eval_cell_type_order)), eval_cell_type_order, rotation=45)
    plt.yticks(np.arange(len(eval_cell_type_order)), eval_cell_type_order)
    plt.savefig(output_path + "/prior_var.png")
    plt.close()

    np.save(output_path + "/Estimated_prior_var.npy", eval_prior_KK_var)

    plt.imshow(tools.linear_transform(eval_KK_strength, GT_strength_connectivity), cmap='RdBu_r', interpolation="nearest", vmin=vmin_KK, vmax=vmax_KK)
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
    # plt.imshow(eval_KK_prob, interpolation="nearest", cmap='bone')
    # plt.colorbar()
    # plt.xlabel("Pre")
    # plt.ylabel("Post")
    # plt.title("eval_KK_prob, corr = " + str(corr_prob_KK)[:7] + ", spearman = " + str(spearman_corr_prob_KK)[:7])
    # plt.xticks(np.arange(len(eval_cell_type_order)), eval_cell_type_order, rotation=45)
    # plt.yticks(np.arange(len(eval_cell_type_order)), eval_cell_type_order)
    # plt.savefig(output_path + "/prob.png")
    # plt.close()

    # np.save(output_path + "/Estimated_prob.npy", eval_KK_prob)

    # plot
    plt.imshow(eval_prior_KK_strength, cmap='RdBu_r', interpolation="nearest")
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
    # plt.imshow(eval_prior_KK_prob, interpolation="nearest", cmap='bone')
    # plt.colorbar()
    # plt.xlabel("Pre")
    # plt.ylabel("Post")
    # plt.title("eval_prior_KK_prob, corr = " + str(corr_prob_prior_KK)[:7] + ", spearman = " + str(spearman_corr_prob_prior_KK)[:7])
    # plt.xticks(np.arange(len(eval_cell_type_order)), eval_cell_type_order, rotation=45)
    # plt.yticks(np.arange(len(eval_cell_type_order)), eval_cell_type_order)
    # plt.savefig(output_path + "/prior_prob.png")
    # plt.close()

    # plot ground truth
    plt.imshow(GT_strength_connectivity, cmap='RdBu_r', interpolation="nearest")
    plt.colorbar()
    plt.xlabel("Pre")
    plt.ylabel("Post")
    plt.title("Ground truth strength connectivity")
    plt.xticks(np.arange(len(eval_cell_type_order)), eval_cell_type_order, rotation=45)
    plt.yticks(np.arange(len(eval_cell_type_order)), eval_cell_type_order)
    plt.savefig(output_path + "/GT_strength.png")
    plt.close()

    # plt.imshow(GT_prob_connectivity, interpolation="nearest", cmap='bone')
    # plt.colorbar()
    # plt.xlabel("Pre")
    # plt.ylabel("Post")
    # plt.title("Ground truth prob connectivity")
    # plt.xticks(np.arange(len(eval_cell_type_order)), eval_cell_type_order, rotation=45)
    # plt.yticks(np.arange(len(eval_cell_type_order)), eval_cell_type_order)
    # plt.savefig(output_path + "/GT_prob.png")
    # plt.close()


    ############## Attention KK 3
    # plt.imshow(eval_KK_3_strength, cmap='RdBu_r', interpolation="nearest")
    # plt.colorbar()
    # plt.xlabel("Pre")
    # plt.ylabel("Post")
    # plt.title("eval_KK_3_strength, corr = " + str(corr_strength_KK_3)[:7] + ", spearman = " + str(spearman_corr_strength_KK_3)[:7])
    # plt.xticks(np.arange(len(eval_cell_type_order)), eval_cell_type_order, rotation=45)
    # plt.yticks(np.arange(len(eval_cell_type_order)), eval_cell_type_order)
    # plt.savefig(output_path + "/strength_3.png")
    # plt.close()