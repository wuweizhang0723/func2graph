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
from sklearn.metrics import r2_score

from func2graph import data, models, baselines, tools


if __name__ == "__main__":
    # Parse arguments add all hyperparameters to the parser
    parser = argparse.ArgumentParser(description="Model Hyperparameters")

    # "GLM_M", "GLM_sim"
    parser.add_argument(
        "--model_type", type=str, default="GLM_M", help="Model type"
    )
    parser.add_argument("--out_folder", help="the output folder")

    # Data

    parser.add_argument("--input_mouse")
    parser.add_argument("--input_sessions")

    parser.add_argument("--k", default=1)
    parser.add_argument("--batch_size", help="the batch size", type=int, default=32)
    parser.add_argument("--normalization", default="session")   # "none" or "session" or "neuron" or "log"

    # Model

    parser.add_argument("--model_random_seed", default=42)
    parser.add_argument("--learning_rate", help="learning rate", default=1e-4)
    parser.add_argument("--scheduler", default="plateau")    # "none" or "plateau"
    parser.add_argument("--weight_decay", default=0)

    args = parser.parse_args()

    # Set the hyperparameters
    model_type = args.model_type
    out_folder = args.out_folder

    # Data

    input_mouse = args.input_mouse
    input_sessions = [str(session) for session in args.input_sessions.split('_')]

    k = int(args.k)
    batch_size = int(args.batch_size)
    normalization = args.normalization

    # Model

    model_random_seed = int(args.model_random_seed)
    learning_rate = float(args.learning_rate)
    scheduler = args.scheduler
    weight_decay = float(args.weight_decay)


    output_path = (
        out_folder
        + model_type
        + "_"
        + input_mouse
        + "_"
        + args.input_sessions
        + "_"
        + str(k)
        + "_"
        + str(batch_size)
        + "_"
        + normalization
        + "_"
        + str(model_random_seed)
        + "_"
        + str(learning_rate)
        + "_"
        + scheduler
        + "_"
        + str(weight_decay)
    )

    train_dataloader_list, val_dataloader_list, num_unqiue_neurons, cell_type_order, all_sessions_new_cell_type_id, sessions_2_original_cell_type, neuron_id_2_cell_type_id = data.generate_mouse_all_sessions_data_for_GLM(
        input_mouse=input_mouse,
        input_sessions=input_sessions,
        k=k, 
        batch_size=batch_size, 
        normalization=normalization
    )

    # Make the ground truth connectivity matrix ----------------------------------------------------------
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

    if model_type == "GLM_M":

        all_sessions_avg_NN = []   # len is the number of sessions, each session has an avg NN matrix

        for i in range(len(train_dataloader_list)):   # loop over all sessions
            checkpoint_path = output_path + "/checkpoint" + str(i)
            log_path = out_folder + "/log"

            train_dataloader = train_dataloader_list[i]
            val_dataloader = val_dataloader_list[i]

            num_neurons = len(all_sessions_new_cell_type_id[i])
            single_model = baselines.GLM_M(
                num_neurons=num_neurons,
                k=k,
                learning_rate=learning_rate,
                scheduler=scheduler,
                weight_decay=weight_decay,
                model_random_seed=model_random_seed,
            )



            # NN = single_model.W_list[0].weight.data.cpu().detach().numpy()
            # KK_strength = tools.multisession_NN_to_KK_1(
            #     [NN],
            #     None,
            #     cell_type_order,
            #     [all_sessions_new_cell_type_id[i]],
            # )
            # eval_KK_strength = tools.experiment_KK_to_eval_KK(KK_strength, cell_type_order, eval_cell_type_order)

            # corr = stats.pearsonr(GT_strength_connectivity.flatten(), eval_KK_strength.flatten())[0]

            # plt.imshow(eval_KK_strength, interpolation="nearest")
            # plt.colorbar()
            # plt.savefig(out_folder + "/default.png")
            # plt.close()





            es = EarlyStopping(monitor="VAL_mse_loss", patience=20)  ###########
            checkpoint_callback = ModelCheckpoint(
                checkpoint_path, monitor="VAL_mse_loss", mode="min", save_top_k=1
            )

            lr_monitor = LearningRateMonitor()
            logger = TensorBoardLogger(log_path, name="model" + str(i))
            trainer = pl.Trainer(
                devices=[0],
                accelerator="gpu",
                callbacks=[es, checkpoint_callback, lr_monitor],
                benchmark=False,
                profiler="simple",
                logger=logger,
                max_epochs=500,
                gradient_clip_val=0,
            )
            trainer.fit(single_model, train_dataloader, val_dataloader)

            ############################################################################################################
            # Evaluate the model --------------------------------------------------------------------------------
            ############################################################################################################

            model_checkpoint_path = checkpoint_path + "/" + listdir(checkpoint_path)[-1]
            trained_model = single_model.load_from_checkpoint(model_checkpoint_path)
            trained_model.eval()

            # Get validation prediction ############################################################
            val_results = trainer.predict(single_model, dataloaders=[val_dataloader], ckpt_path=model_checkpoint_path)
            val_results = torch.cat(val_results, dim=0).cpu().detach().numpy()
            val_pred = val_results[:,:num_neurons]     # (num_samples, num_neurons)
            val_target = val_results[:,num_neurons:]

            corr = stats.pearsonr(val_pred.flatten(), val_target.flatten())[0]
            R_squared = r2_score(val_pred.flatten(), val_target.flatten())
            plt.scatter(val_pred.flatten(), val_target.flatten(), s=1)
            plt.xlabel("Predicted")
            plt.ylabel("Target")
            plt.title("Validation, corr = " + str(corr)[:7] + ", R^2 = " + str(R_squared)[:7])
            plt.savefig(checkpoint_path + "/scatter.png")
            plt.close()

            for neuron in range(10):
                plt.subplot(10, 1, neuron+1)
                plt.plot(val_pred[:100, neuron], label="Prediction")
                plt.plot(val_target[:100, neuron], label="Ground Truth")
            plt.legend()
            plt.savefig(checkpoint_path + "/curve.png")
            plt.close()

            # Get the k NN matrix ############################################################
            NN_list = []   # should contrain k NN matrices
            KK_list = []   # should contrain k kk matrices
            for j in range(k):
                NN = trained_model.W_list[j].weight.data.cpu().detach().numpy()
                NN_list.append(NN)

                KK_strength = tools.multisession_NN_to_KK_1(
                    [NN], 
                    None,
                    cell_type_order,
                    [all_sessions_new_cell_type_id[i]],
                )
                eval_KK_strength = tools.experiment_KK_to_eval_KK(KK_strength, cell_type_order, eval_cell_type_order)

                KK_list.append(eval_KK_strength)
                corr = stats.pearsonr(GT_strength_connectivity.flatten(), eval_KK_strength.flatten())[0]
                spearman = stats.spearmanr(GT_strength_connectivity.flatten(), eval_KK_strength.flatten())[0]

                plt.imshow(eval_KK_strength, interpolation="nearest")
                plt.colorbar()
                plt.xlabel("Pre")
                plt.ylabel("Post")
                plt.title("KK_strength_" + str(j) + ", corr = " + str(corr)[:7] + ", spearman = " + str(spearman)[:7])
                plt.xticks(np.arange(len(eval_cell_type_order)), eval_cell_type_order, rotation=45)
                plt.yticks(np.arange(len(eval_cell_type_order)), eval_cell_type_order)
                plt.savefig(checkpoint_path + "/KK_strength_" + str(j) + ".png")
                plt.close()

            avg_NN = np.mean(NN_list, axis=0)

            all_sessions_avg_NN.append(avg_NN)

        ############################################################################################################
        ### multisession N*N => ONE K*K
        ###
        ### Use functions in tools.py
        ############################################################################################################

        experiment_KK_strength = tools.multisession_NN_to_KK_1(
            all_sessions_avg_NN, 
            None,
            cell_type_order,
            all_sessions_new_cell_type_id,
        )

        eval_KK_strength = tools.experiment_KK_to_eval_KK(experiment_KK_strength, cell_type_order, eval_cell_type_order)

        # get correlation
        corr_strength_KK = stats.pearsonr(GT_strength_connectivity.flatten(), eval_KK_strength.flatten())[0]
        spearman_corr_strength_KK = stats.spearmanr(GT_strength_connectivity.flatten(), eval_KK_strength.flatten())[0]

        ############################################################ plot
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