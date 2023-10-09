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
from os import listdir

from func2graph import data, models, baselines



if __name__ == "__main__":
    # Parse arguments add all hyperparameters to the parser
    parser = argparse.ArgumentParser(description="Model Hyperparameters")

    # "Attention_Autoencoder" or "Baseline_2"
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

    parser.add_argument("--weight_scale", default=1)
    parser.add_argument("--init_scale", default=1)

    parser.add_argument("--total_time", help="total time", default=30000)
    parser.add_argument("--data_random_seed", help="data random seed", default=42)

    parser.add_argument("--weight_type", default="random")    # "random"

    parser.add_argument("--train_data_size", default=20000)
    parser.add_argument("--window_size", default=200)

    parser.add_argument("--batch_size", help="the batch size", type=int, default=32)

    parser.add_argument("--task_type", default="reconstruction")  # "reconstruction" or "prediction" or "baseline_2"
    parser.add_argument("--predict_window_size", default=100)

    parser.add_argument("--data_type", default="wuwei")   # "ziyu"

    # Model
    parser.add_argument("--model_random_seed", default=42)

    parser.add_argument("--hidden_size_1", help="hidden size 1", default=128)
    parser.add_argument("--h_layers_1", help="h layers 1", default=2)
    
    parser.add_argument("--hidden_size_1_S", help="hidden size 1", default=128)
    parser.add_argument("--hidden_size_1_T", help="hidden size 1", default=128)

    parser.add_argument("--heads", help="heads", default=1)
    parser.add_argument("--attention_layers", help="attention layers", default=1)
    parser.add_argument("--dim_key", default=64)

    parser.add_argument("--hidden_size_2", help="hidden size 2", default=258)
    parser.add_argument("--h_layers_2", help="h layers 2", default=2)

    parser.add_argument("--dropout", help="dropout", default=0.2)

    parser.add_argument("--learning_rate", help="learning rate", default=1e-4)

    parser.add_argument("--pos_enc_type", default="none")

    parser.add_argument("--attention_type", default="spatial_temporal_1")  # "spatial_temporal_1" or "spatial_temporal_2" or "spatial_temporal_3" or "spatial"

    # Baseline_2



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

    train_data_size = int(args.train_data_size)
    window_size = int(args.window_size)

    batch_size = int(args.batch_size)

    task_type = args.task_type
    predict_window_size = int(args.predict_window_size)

    data_type = args.data_type

    # Model
    model_random_seed = int(args.model_random_seed)

    hidden_size_1 = int(args.hidden_size_1)
    h_layers_1 = int(args.h_layers_1)
    hidden_size_1_S = int(args.hidden_size_1_S)
    hidden_size_1_T = int(args.hidden_size_1_T)

    heads = int(args.heads)
    attention_layers = int(args.attention_layers)
    dim_key = int(args.dim_key)

    hidden_size_2 = int(args.hidden_size_2)
    h_layers_2 = int(args.h_layers_2)

    dropout = float(args.dropout)

    learning_rate = float(args.learning_rate)

    pos_enc_type = args.pos_enc_type

    attention_type = args.attention_type



    output_path = (
        out_folder
        + model_type
        + "_"
        + data_type
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
        + str(data_random_seed)
        + "_"
        + weight_type
        + "_"
        + str(train_data_size)
        + "_"
        + str(window_size)
        + "_"
        + task_type
        + "_"
        + str(predict_window_size)
        + "_"
        + str(model_random_seed)
        + "_"
        + str(hidden_size_1)
        + "_"
        + str(hidden_size_1_S)
        + "_"
        + str(hidden_size_1_T)
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
        + str(learning_rate)
        + "_"
        + pos_enc_type
        + "_"
        + attention_type
    )

    checkpoint_path = output_path
    log_path = out_folder + "/log"

    
    trainloader, validloader, weight_matrix = data.generate_simulation_data(
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
        train_data_size=train_data_size,
        window_size=window_size,
        batch_size=batch_size,
        task_type=task_type,
        predict_window_size=predict_window_size,
        data_type=data_type,
    )
    if data_type == "wuwei" or data_type == "ziyu":
        weight_matrix = weight_matrix.detach().numpy()
    elif data_type == "c_elegans":
        weight_matrix_E = weight_matrix[0].detach().numpy()
        weight_matrix_Chem = weight_matrix[1].detach().numpy()

    if model_type == "Attention_Autoencoder":
        single_model = models.Attention_Autoencoder(
            model_random_seed=model_random_seed,
            neuron_num=neuron_num,
            window_size=window_size,
            hidden_size_1=hidden_size_1,
            h_layers_1=h_layers_1,
            heads=heads,
            attention_layers=attention_layers,
            dim_key=dim_key,
            hidden_size_2=hidden_size_2,
            h_layers_2=h_layers_2,
            dropout=dropout,
            learning_rate=learning_rate,
            pos_enc_type=pos_enc_type,
            task_type=task_type,
            predict_window_size=predict_window_size,
        )
    elif model_type == "Baseline_2":
        single_model = baselines.Baseline_2(
            neuron_num=neuron_num,
            learning_rate=learning_rate,
            simulated_network_type=2,
            model_random_seed=model_random_seed,
        )
    elif model_type == "Spatial_Temporal_Attention_Model":
        single_model = models.Spatial_Temporal_Attention_Model(
            model_random_seed=model_random_seed,
            neuron_num=neuron_num,
            window_size=window_size,
            predict_window_size = predict_window_size,
            hidden_size_1_T = hidden_size_1_T,
            hidden_size_1_S = hidden_size_1_S,
            h_layers_1=h_layers_1,
            attention_type = attention_type,
            pos_enc_type = pos_enc_type,
            heads=heads,
            hidden_size_2=hidden_size_2,
            h_layers_2=h_layers_2,
            dropout=dropout,
            learning_rate=learning_rate,
            task_type=task_type,
        )


    es = EarlyStopping(monitor="val_loss", patience=12)  ###########
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
        max_epochs=500,
    )

    trainer.fit(single_model, trainloader, validloader)




    # Add evaluation after training：
    # 1.ground-truth W plot 
    # 2.estimated W plot (train & val)
    # 3.validation loss 
    # 4.activity prediction plot

    if model_type == "Attention_Autoencoder" or model_type == "Spatial_Temporal_Attention_Model":
        if model_type == "Attention_Autoencoder":
            predict_mode_model = models.Attention_Autoencoder(
                model_random_seed=model_random_seed,
                neuron_num=neuron_num,
                window_size=window_size,
                hidden_size_1=hidden_size_1,
                h_layers_1=h_layers_1,
                heads=heads,
                attention_layers=attention_layers,
                dim_key=dim_key,
                hidden_size_2=hidden_size_2,
                h_layers_2=h_layers_2,
                dropout=dropout,
                learning_rate=learning_rate,
                pos_enc_type=pos_enc_type,
                task_type=task_type,
                predict_window_size=predict_window_size,
                prediction_mode=True,
            )
        elif model_type == "Spatial_Temporal_Attention_Model":
            predict_mode_model = models.Spatial_Temporal_Attention_Model(
                model_random_seed=model_random_seed,
                neuron_num=neuron_num,
                window_size=window_size,
                predict_window_size = predict_window_size,
                hidden_size_1_T=hidden_size_1_T,
                hidden_size_1_S=hidden_size_1_S,
                h_layers_1=h_layers_1,
                attention_type = attention_type,
                pos_enc_type = pos_enc_type,
                heads=heads,
                hidden_size_2=hidden_size_2,
                h_layers_2=h_layers_2,
                dropout=dropout,
                learning_rate=learning_rate,
                task_type=task_type,
                prediction_mode=True,
            )

        model_checkpoint_path = checkpoint_path + "/" + listdir(checkpoint_path)[-1]

        train_results = trainer.predict(predict_mode_model, dataloaders=[trainloader], ckpt_path=model_checkpoint_path)

        predictions = []
        ground_truths = []
        attentions = []
        for i in range(len(train_results)):
            x_hat = train_results[i][0]    # batch_size * (neuron_num*time)
            x = train_results[i][1]
            attention = train_results[i][2]
            attention = attention.view(-1, neuron_num, neuron_num)

            predictions.append(x_hat)
            ground_truths.append(x)
            attentions.append(attention)
        
        predictions = torch.cat(predictions, dim=0).cpu().numpy()  # N * neuron_num * window_size
        ground_truths = torch.cat(ground_truths, dim=0).cpu().numpy()  # N * neuron_num * window_size
        attentions = torch.cat(attentions, dim=0).cpu().numpy()    # N * neuron_num * neuron_num
        
        # # get average attention across 
        avg_attention = np.mean(attentions, axis=0)   # neuron_num * neuron_num
        W = avg_attention
        

    elif model_type == "Baseline_2":
        predict_mode_model = baselines.Baseline_2(
            neuron_num=neuron_num,
            learning_rate=learning_rate,
            simulated_network_type=2,
            model_random_seed=model_random_seed,
        )
        model_checkpoint_path = checkpoint_path + "/" + listdir(checkpoint_path)[-1]

        model_checkpoint = predict_mode_model.load_from_checkpoint(model_checkpoint_path)
        model_checkpoint.eval()
        W = model_checkpoint.W.weight.data
        W = W.cpu().detach().numpy()


    if data_type == "wuwei" or data_type == "ziyu":
        estimation_corr = np.corrcoef(W.flatten(), weight_matrix.flatten())[0, 1]
        estimation_corr_abs = np.corrcoef(W.flatten(), np.abs(weight_matrix).flatten())[0, 1]

        plt.imshow(W)
        plt.colorbar()
        plt.title("Estimated W" + " (corr: " + str(estimation_corr)[:6] + ") " + " (corr_abs: " + str(estimation_corr_abs)[:6] + ")")
        plt.savefig(output_path + "/Estimated_W.png")
        plt.close()

        plt.imshow(weight_matrix)
        plt.colorbar()
        plt.title("Ground-truth W")
        plt.savefig(output_path + "/Ground_truth_W.png")
        plt.close()

        # save estimated W to npy file
        np.save(output_path + "/" + "Estimated_W_" + str(model_random_seed) + ".npy", W)
    
    elif data_type == "c_elegans":
        estimation_corr_E = np.corrcoef(W.flatten(), weight_matrix_E.flatten())[0, 1]
        estimation_corr_Chem = np.corrcoef(W.flatten(), weight_matrix_Chem.flatten())[0, 1]

        estimation_corr_E_abs = np.corrcoef(np.abs(W).flatten(), weight_matrix_E.flatten())[0, 1]
        estimation_corr_Chem_abs = np.corrcoef(np.abs(W).flatten(), weight_matrix_Chem.flatten())[0, 1]

        plt.imshow(W)
        plt.colorbar()
        plt.title("Estimated W" + " (corr_E: " + str(estimation_corr_E)[:6] + ") " + " (corr_Chem: " + str(estimation_corr_Chem)[:6] + ")")
        plt.savefig(output_path + "/Estimated_W.png")
        plt.close()

        plt.imshow(np.abs(W))
        plt.colorbar()
        plt.title("Estimated W" + " (corr_E_abs: " + str(estimation_corr_E_abs)[:6] + ") " + " (corr_Chem_abs: " + str(estimation_corr_Chem_abs)[:6] + ")")
        plt.savefig(output_path + "/Estimated_W_abs.png")
        plt.close()

        plt.imshow(weight_matrix_E)
        plt.colorbar()
        plt.title("Ground-truth W_E")
        plt.savefig(output_path + "/Ground_truth_W_E.png")
        plt.close()

        plt.imshow(weight_matrix_Chem)
        plt.colorbar()
        plt.title("Ground-truth W_Chem")
        plt.savefig(output_path + "/Ground_truth_W_Chem.png")
        plt.close()

        # save estimated W to npy file
        np.save(output_path + "/" + "Estimated_W_" + str(model_random_seed) + ".npy", W)