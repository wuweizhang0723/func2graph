import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
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
from torchmetrics import AUROC

from func2graph import data, models, baselines, tools


if __name__ == "__main__":
    # Parse arguments add all hyperparameters to the parser
    parser = argparse.ArgumentParser(description="Model Hyperparameters")

    parser.add_argument(
        "--model_type", type=str, default="AwC_sim", help="Model type"
    )
    parser.add_argument("--out_folder", help="the output folder")

    # Data
    parser.add_argument("--neuron_num", help="the number of neurons", type=int, default=10)
    parser.add_argument("--dt", help="dt", default=0.001)
    parser.add_argument("--tau", help="tau", default=0.3)
    parser.add_argument("--spike_neuron_num", default=2)
    parser.add_argument("--spike_input", default=5)

    parser.add_argument("--weight_scale", default=0.1)
    parser.add_argument("--init_scale", default=0.1)

    parser.add_argument("--total_time", help="total time", default=30000)
    parser.add_argument("--data_random_seed", help="data random seed", default=42)

    parser.add_argument("--weight_type", default="cell_type")    # "random"
    parser.add_argument("--window_size", default=200)
    parser.add_argument("--batch_size", help="the batch size", type=int, default=32)

    parser.add_argument("--task_type", default="prediction")  # "reconstruction" or "prediction" or "baseline_2"
    parser.add_argument("--predict_window_size", default=100)

    parser.add_argument("--data_type", default="wuwei")   # "ziyu"
    parser.add_argument("--spatial_partial_measurement", default=200)   # between 0 and neuron_num

    # Model
    parser.add_argument("--model_random_seed", default=42)

    parser.add_argument("--attention_layers", help="attention layers", default=1)
    parser.add_argument("--attention_activation", default="softmax")    # "softmax" or "sigmoid" or "tanh" or "none"
    parser.add_argument("--pos_enc_type", default="lookup_table")

    parser.add_argument("--causal_temporal_map", default='none')   # 'none', 'off_diagonal_1', 'off_diagonal', 'lower_triangle'
    parser.add_argument("--causal_temporal_map_diff", default=1)   # 1 or 2 or 3, ...
    parser.add_argument("--l1_on_causal_temporal_map", default=0)   # alpha penalty

    parser.add_argument("--learning_rate", help="learning rate", default=1e-4)
    parser.add_argument("--scheduler", default="plateau")    # "none" or "plateau"

    parser.add_argument("--loss_function", default="mse")   # "mse" or "poisson" or "gaussian"
    parser.add_argument("--weight_decay", default=0)
    parser.add_argument("--constraint_var", default=1)
    parser.add_argument("--constraint_loss_weight", default=0)


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
    data_random_seed = int(args.data_random_seed)

    weight_type = args.weight_type
    window_size = int(args.window_size)
    batch_size = int(args.batch_size)

    task_type = args.task_type
    predict_window_size = int(args.predict_window_size)

    data_type = args.data_type
    spatial_partial_measurement = int(args.spatial_partial_measurement)

    # Model

    model_random_seed = int(args.model_random_seed)

    attention_layers = int(args.attention_layers)
    attention_activation = args.attention_activation
    pos_enc_type = args.pos_enc_type

    causal_temporal_map = args.causal_temporal_map
    causal_temporal_map_diff = int(args.causal_temporal_map_diff)
    l1_on_causal_temporal_map = float(args.l1_on_causal_temporal_map)

    learning_rate = float(args.learning_rate)
    scheduler = args.scheduler

    loss_function = args.loss_function
    weight_decay = float(args.weight_decay)
    constraint_var = float(args.constraint_var)
    constraint_loss_weight = float(args.constraint_loss_weight)


    output_path = (
        out_folder
        + model_type
        + "_"
        + str(data_type)
        + "_"
        + str(neuron_num)
        + "_"
        + str(spatial_partial_measurement)
        + "_"
        + str(data_random_seed)
        + "_"
        + str(model_random_seed)
        + "_"
        + str(attention_layers)
        + "_"
        + str(attention_activation)
        + "_"
        + str(pos_enc_type)
        + "_"
        + str(causal_temporal_map)
        + "_"
        + str(causal_temporal_map_diff)
        + "_"
        + str(l1_on_causal_temporal_map)
        + "_"
        + str(learning_rate)
        + "_"
        + str(scheduler)
        + "_"
        + str(loss_function)
        + "_"
        + str(weight_decay)
        + "_"
        + str(constraint_var)
        + "_"
        + str(constraint_loss_weight)
    )

    checkpoint_path = output_path
    log_path = out_folder + "/log"

    data_result = data.generate_simulation_data(
        neuron_num=neuron_num,
        dt=dt,
        tau=tau,
        spike_neuron_num=spike_neuron_num,
        spike_input=spike_input,
        weight_scale=weight_scale,
        init_scale=init_scale,
        total_time=total_time,
        data_random_seed=data_random_seed,
        weight_type=weight_type,
        window_size=window_size,
        batch_size=batch_size,
        task_type=task_type,
        predict_window_size=predict_window_size,
        data_type=data_type,
        spatial_partial_measurement=spatial_partial_measurement,
    )
    if data_type == "wuwei" or data_type == "ziyu":
        # cell_type_ids records the cell type of each neuron
        trainloader, validloader, weight_matrix, cell_type_ids, cell_type_order, cell_type_count = data_result
        weight_matrix = weight_matrix.detach().numpy()
    elif data_type == "c_elegans":
        trainloader, validloader, weight_matrix = data_result
        weight_matrix_E = weight_matrix[0].detach().numpy()
        weight_matrix_Chem = weight_matrix[1].detach().numpy()

    # for spatial_partial_measurement !!!!!!!!!!!!!!!!!!!
    if spatial_partial_measurement != neuron_num:
        neuron_num = spatial_partial_measurement

    single_model = models.Attention_With_Constraint_sim(
        model_random_seed=model_random_seed,
        neuron_num=neuron_num,
        num_cell_types=4,
        window_size=window_size,
        attention_layers=attention_layers,
        learning_rate=learning_rate,
        scheduler=scheduler,
        pos_enc_type=pos_enc_type,
        task_type=task_type,
        predict_window_size=predict_window_size,
        loss_function = loss_function,
        attention_activation = attention_activation,
        weight_decay = weight_decay,
        causal_temporal_map = causal_temporal_map,
        causal_temporal_map_diff = causal_temporal_map_diff,
        l1_on_causal_temporal_map = l1_on_causal_temporal_map,
        constraint_loss_weight = constraint_loss_weight,
        constraint_var = constraint_var
    )

    es = EarlyStopping(monitor="VAL_sum_loss", patience=20)  ###########
    checkpoint_callback = ModelCheckpoint(
        checkpoint_path, monitor="VAL_sum_loss", mode="min", save_top_k=1
    )
    lr_monitor = LearningRateMonitor()
    logger = TensorBoardLogger(log_path, name="model")
    trainer = pl.Trainer(
        devices=[3],
        accelerator="gpu",
        callbacks=[es, checkpoint_callback, lr_monitor],
        benchmark=False,
        profiler="simple",
        logger=logger,
        max_epochs=1000,
    )

    trainer.fit(single_model, trainloader, validloader)


    # evaluate the model -----------------------------------------------------

    model_checkpoint_path = checkpoint_path + "/" + listdir(checkpoint_path)[-1]

    train_results = trainer.predict(single_model, dataloaders=[trainloader], ckpt_path=model_checkpoint_path)

    predictions = []
    ground_truths = []
    attentions = []
    for i in range(len(train_results)):
        x_hat = train_results[i][0]    # batch_size * (neuron_num*time)
        x = train_results[i][1]
        attention = train_results[i][2]
        # neuron_embedding = train_results[i][3]
        
        attention = attention.view(-1, neuron_num, neuron_num)

        predictions.append(x_hat)
        ground_truths.append(x)
        attentions.append(attention)

    predictions = torch.cat(predictions, dim=0).cpu().numpy()  # N * neuron_num * window_size
    ground_truths = torch.cat(ground_truths, dim=0).cpu().numpy()  # N * neuron_num * window_size
    attentions = torch.cat(attentions, dim=0).cpu().numpy()    # N * neuron_num * neuron_num

    avg_attention = np.mean(attentions, axis=0)   # neuron_num * neuron_num
    W = avg_attention


    if data_type == "wuwei" or data_type == "ziyu":

        ############################################################# Strength connection evaluation 

        estimation_corr = np.corrcoef(W.flatten(), weight_matrix.flatten())[0, 1]
        estimation_corr_abs = np.corrcoef(np.abs(W.flatten()), np.abs(weight_matrix).flatten())[0, 1]

        strength_matrix = np.zeros((4, 4))
        strength_matrix[0, 0] = 0.3
        strength_matrix[1, 0] = 0.59
        strength_matrix[2, 0] = 0.88
        strength_matrix[3, 0] = 1.89

        strength_matrix[0, 1] = -0.43
        strength_matrix[1, 1] = -0.53
        strength_matrix[2, 1] = -0.60
        strength_matrix[3, 1] = -0.44

        strength_matrix[0, 2] = -0.31
        strength_matrix[1, 2] = -0.43
        strength_matrix[2, 2] = -0.43
        strength_matrix[3, 2] = -0.79

        strength_matrix[0, 3] = -0.25
        strength_matrix[1, 3] = -0.30
        strength_matrix[2, 3] = -0.42
        strength_matrix[3, 3] = -0.33

        cell_type_id2cell_type = {0:'EC', 1:'Pv', 2:'Sst', 3:'Vip'}

        KK_strength = tools.calculate_cell_type_level_connectivity_matrix_remove_no_connection(
            connectivity_matrix_new=W,
            connectivity_matrix_GT=weight_matrix, 
            cell_type_id2cell_type=cell_type_id2cell_type,
            cell_type_count=cell_type_count
        )
        corr_strength_KK = np.corrcoef(KK_strength.flatten(), strength_matrix.flatten())[0, 1]
        spearman_corr_strength_KK = stats.spearmanr(KK_strength.flatten(), strength_matrix.flatten())[0]

        ############################################################# Binary connection evaluation

        cutoff_matrix = np.zeros((4, 4))
        cutoff_matrix[0, 0] = 13/229
        cutoff_matrix[1, 0] = 22/53
        cutoff_matrix[2, 0] = 20/67
        cutoff_matrix[3, 0] = 11/68

        cutoff_matrix[0, 1] = 18/52
        cutoff_matrix[1, 1] = 45/114
        cutoff_matrix[2, 1] = 8/88
        cutoff_matrix[3, 1] = 0/54

        cutoff_matrix[0, 2] = 13/56
        cutoff_matrix[1, 2] = 15/84
        cutoff_matrix[2, 2] = 8/154
        cutoff_matrix[3, 2] = 25/84

        cutoff_matrix[0, 3] = 3/62
        cutoff_matrix[1, 3] = 1/54
        cutoff_matrix[2, 3] = 12/87
        cutoff_matrix[3, 3] = 2/209

        # make all nonzero elements in GT weight matrix to 1
        binary_GT = np.zeros(weight_matrix.shape)
        binary_GT[weight_matrix != 0] = 1
        # min max normalization
        prob_W = np.abs(W)
        prob_W = (prob_W - np.min(prob_W)) / (np.max(prob_W) - np.min(prob_W))

        cross_entropy = F.binary_cross_entropy(torch.from_numpy(prob_W).float().view(-1), torch.from_numpy(binary_GT).float().view(-1))
        auroc = AUROC(task="binary")
        auroc_val = auroc(torch.from_numpy(prob_W).float().view(-1), torch.from_numpy(binary_GT).float().view(-1))

        # convert n*n prob_W to k*k prob_W
        KK_prob = tools.calculate_cell_type_level_connectivity_matrix(
            connectivity_matrix_new=prob_W,
            cell_type_id2cell_type=cell_type_id2cell_type,
            cell_type_count=cell_type_count
        )

        corr_prob_KK = np.corrcoef(KK_prob.flatten(), cutoff_matrix.flatten())[0, 1]
        spearman_corr_prob_KK = stats.spearmanr(KK_prob.flatten(), cutoff_matrix.flatten())[0]

        ############################################################# Constraint prior evaluation
        
        trained_model = single_model.load_from_checkpoint(model_checkpoint_path)
        trained_model.eval()

        prior_KK_strength = trained_model.cell_type_level_constraint.cpu().detach().numpy()
        prior_KK_prob = np.abs(prior_KK_strength)
        prior_KK_prob = (prior_KK_prob - np.min(prior_KK_prob)) / (np.max(prior_KK_prob) - np.min(prior_KK_prob))

        corr_strength_prior_KK = stats.pearsonr(strength_matrix.flatten(), prior_KK_strength.flatten())[0]
        spearman_corr_strength_prior_KK = stats.spearmanr(strength_matrix.flatten(), prior_KK_strength.flatten())[0]

        corr_prob_prior_KK = stats.pearsonr(cutoff_matrix.flatten(), prior_KK_prob.flatten())[0]
        spearman_corr_prob_prior_KK = stats.spearmanr(cutoff_matrix.flatten(), prior_KK_prob.flatten())[0]


        ############################################################# plot

        plt.imshow(KK_strength, interpolation="nearest")
        plt.colorbar()
        plt.xlabel("Pre")
        plt.ylabel("Post")
        plt.title("KK_strength, corr = " + str(corr_strength_KK)[:7] + ", spearman = " + str(spearman_corr_strength_KK)[:7])
        plt.savefig(output_path + "/KK_strength.png")
        plt.close()

        np.save(output_path + "/Estimated_KK_strength.npy", KK_strength)

        plt.imshow(KK_prob, interpolation="nearest", cmap='bone')
        plt.colorbar()
        plt.xlabel("Pre")
        plt.ylabel("Post")
        plt.title("KK_prob, corr = " + str(corr_prob_KK)[:7] + ", spearman = " + str(spearman_corr_prob_KK)[:7])
        plt.savefig(output_path + "/KK_prob.png")
        plt.close()

        np.save(output_path + "/Estimated_KK_prob.npy", KK_prob)

        plt.imshow(prior_KK_strength, interpolation="nearest")
        plt.colorbar()
        plt.xlabel("Pre")
        plt.ylabel("Post")
        plt.title("eval_prior_KK_strength, corr = " + str(corr_strength_prior_KK)[:7] + ", spearman = " + str(spearman_corr_strength_prior_KK)[:7])
        plt.savefig(output_path + "/prior_strength.png")
        plt.close()

        np.save(output_path + "/Estimated_prior_strength.npy", prior_KK_strength)

        plt.imshow(prior_KK_prob, interpolation="nearest", cmap='bone')
        plt.colorbar()
        plt.xlabel("Pre")
        plt.ylabel("Post")
        plt.title("eval_prior_KK_prob, corr = " + str(corr_prob_prior_KK)[:7] + ", spearman = " + str(spearman_corr_prob_prior_KK)[:7])
        plt.savefig(output_path + "/prior_prob.png")
        plt.close()

        np.save(output_path + "/Estimated_prior_prob.npy", prior_KK_prob)


        # NN
        plt.imshow(W)
        plt.colorbar()
        plt.title("W" + " (corr: " + str(estimation_corr)[:6] + ") " + " (corr_abs: " + str(estimation_corr_abs)[:6] + ")")
        plt.savefig(output_path + "/NN_strength.png")
        plt.close()

        plt.imshow(prob_W, cmap='bone')
        plt.colorbar()
        plt.title("NN_prob" + " (BCE: " + str(cross_entropy.numpy())[:6] + ") " + " (AUROC: " + str(auroc_val.numpy())[:6] + ")")
        plt.savefig(output_path + "/NN_prob.png")
        plt.close()


        # GT
        plt.imshow(strength_matrix)
        plt.colorbar()
        plt.title("GT_(cell_type_level)")
        plt.savefig(output_path + "/GT_KK_strength.png")
        plt.close()

        plt.imshow(cutoff_matrix, cmap='bone')
        plt.colorbar()
        plt.title("prob_GT_(cell_type_level)")
        plt.savefig(output_path + "/GT_KK_prob.png")
        plt.close()

        max_abs = np.max(np.abs(weight_matrix))
        plt.imshow(weight_matrix, cmap='RdBu_r', vmin=-max_abs, vmax=max_abs)
        plt.colorbar()
        plt.title("GT")
        plt.savefig(output_path + "/GT_NN_strength.png")
        plt.close()

        plt.imshow(binary_GT, cmap='bone')
        plt.colorbar()
        plt.title("prob_GT")
        plt.savefig(output_path + "/GT_NN_prob.png")
        plt.close()