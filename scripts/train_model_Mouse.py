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

    # "Attention_Autoencoder" or "Baseline_2"
    parser.add_argument(
        "--model_type", type=str, default="Attention_Autoencoder", help="Model type"
    )
    parser.add_argument("--out_folder", help="the output folder")

    # Data

    parser.add_argument("--window_size", default=200)

    parser.add_argument("--batch_size", help="the batch size", type=int, default=32)

    parser.add_argument("--task_type", default="reconstruction")  # "reconstruction" or "prediction" or "baseline_2" or "mask"
    parser.add_argument("--predict_window_size", default=100)
    parser.add_argument("--mask_size", default=5)

    parser.add_argument("--data_type", default="mouse")

    parser.add_argument("--split_ratio", default=0.8)

    parser.add_argument("--normalization", default="none")   # "none" or "session" or "neuron" or "log"

    # Model

    parser.add_argument("--model_random_seed", default=42)

    parser.add_argument("--hidden_size_1", help="hidden size 1", default=128)
    parser.add_argument("--h_layers_1", help="h layers 1", default=2)
    
    # parser.add_argument("--hidden_size_1_S", help="hidden size 1", default=128)
    # parser.add_argument("--hidden_size_1_T", help="hidden size 1", default=128)

    parser.add_argument("--heads", help="heads", default=1)
    parser.add_argument("--attention_layers", help="attention layers", default=1)
    parser.add_argument("--dim_key", default=64)

    parser.add_argument("--hidden_size_2", help="hidden size 2", default=258)
    parser.add_argument("--h_layers_2", help="h layers 2", default=2)

    parser.add_argument("--dropout", help="dropout", default=0.2)

    parser.add_argument("--learning_rate", help="learning rate", default=1e-4)

    parser.add_argument("--pos_enc_type", default="none")

    parser.add_argument("--loss_function", default="mse")   # "mse" or "poisson" or "gaussian"

    # parser.add_argument("--attention_type", default="spatial_temporal_1")  # "spatial_temporal_1" or "spatial_temporal_2" or "spatial_temporal_3" or "spatial"

    # Baseline_2



    args = parser.parse_args()

    # Set the hyperparameters
    model_type = args.model_type
    out_folder = args.out_folder

    # Data

    window_size = int(args.window_size)

    batch_size = int(args.batch_size)

    task_type = args.task_type
    predict_window_size = int(args.predict_window_size)
    mask_size = int(args.mask_size)

    data_type = args.data_type

    split_ratio = float(args.split_ratio)

    normalization = args.normalization

    if normalization == "log":
        log_input = True
    else:
        log_input = False

    # Model
    model_random_seed = int(args.model_random_seed)

    hidden_size_1 = int(args.hidden_size_1)
    h_layers_1 = int(args.h_layers_1)
    # hidden_size_1_S = int(args.hidden_size_1_S)
    # hidden_size_1_T = int(args.hidden_size_1_T)

    heads = int(args.heads)
    attention_layers = int(args.attention_layers)
    dim_key = int(args.dim_key)

    hidden_size_2 = int(args.hidden_size_2)
    h_layers_2 = int(args.h_layers_2)

    dropout = float(args.dropout)

    learning_rate = float(args.learning_rate)

    pos_enc_type = args.pos_enc_type

    # attention_type = args.attention_type

    loss_function = args.loss_function



    output_path = (
        out_folder
        + model_type
        + "_"
        + data_type
        + "_"
        + normalization
        + "_"
        + str(split_ratio)
        + "_"
        + str(window_size)
        + "_"
        + task_type
        + "_"
        + str(predict_window_size)
        + "_"
        + str(model_random_seed)
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
        + str(hidden_size_1)
        + "_"
        + str(h_layers_1)
        + "_"
        + str(mask_size)
        + "_"
        + loss_function
    )

    checkpoint_path = output_path
    log_path = out_folder + "/log"

    
    trainloader, validloader, weight_matrix, neuron_types = data.generate_simulation_data(
        # train_data_size=train_data_size,
        window_size=window_size,
        batch_size=batch_size,
        split_ratio=split_ratio,
        task_type=task_type,
        predict_window_size=predict_window_size,
        data_type=data_type,
        mask_size=mask_size,
        normalization=normalization,
    )

    neuron_num = len(neuron_types)
    print("neuron_num: ", neuron_num)

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
            loss_function=loss_function,
            log_input=log_input,
        )
    elif model_type == "Baseline_2":
        single_model = baselines.Baseline_2(
            neuron_num=neuron_num,
            learning_rate=learning_rate,
            simulated_network_type=1,
            model_random_seed=model_random_seed,
            loss_function=loss_function,
            log_input=log_input,
        )
    elif model_type == "Spatial_Temporal_Attention_Model":
        single_model = models.Spatial_Temporal_Attention_Model(
            model_random_seed=model_random_seed,
            neuron_num=neuron_num,
            window_size=window_size,
            predict_window_size = predict_window_size,
            # hidden_size_1_T = hidden_size_1_T,
            # hidden_size_1_S = hidden_size_1_S,
            # h_layers_1=h_layers_1,
            # attention_type = attention_type,
            pos_enc_type = pos_enc_type,
            heads=heads,
            hidden_size_2=hidden_size_2,
            h_layers_2=h_layers_2,
            dropout=dropout,
            learning_rate=learning_rate,
            task_type=task_type,
        )


    es = EarlyStopping(monitor=loss_function+" val_loss", patience=20)  ###########
    checkpoint_callback = ModelCheckpoint(
        checkpoint_path, monitor=loss_function+" val_loss", mode="min", save_top_k=1
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

    trainer.fit(single_model, trainloader, validloader)




    # Add evaluation after training： -------------------------------------------------------------
    # 1.ground-truth W plot 
    # 2.estimated W plot (train & val)
    # 3.validation loss 
    # 4.activity prediction plot (scatter plot)

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
                loss_function=loss_function,
                log_input=log_input,
                prediction_mode=True,
            )
        elif model_type == "Spatial_Temporal_Attention_Model":
            predict_mode_model = models.Spatial_Temporal_Attention_Model(
                model_random_seed=model_random_seed,
                neuron_num=neuron_num,
                window_size=window_size,
                predict_window_size = predict_window_size,
                # hidden_size_1_T=hidden_size_1_T,
                # hidden_size_1_S=hidden_size_1_S,
                # h_layers_1=h_layers_1,
                # attention_type = attention_type,
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

            if i % 100 == 0:
                print("Predicting batch: ", i)
        
        predictions = torch.cat(predictions, dim=0).cpu().numpy()  # N * neuron_num * window_size
        ground_truths = torch.cat(ground_truths, dim=0).cpu().numpy()  # N * neuron_num * window_size
        attentions = torch.cat(attentions, dim=0).cpu().numpy()    # N * neuron_num * neuron_num


        # plot scatter plot of prediction vs. ground truth (flatten)
        # plt.scatter(ground_truths.flatten(), predictions.flatten(), s=0.1)
        # plt.xlabel("Ground Truth")
        # plt.ylabel("Prediction")
        # corr = stats.pearsonr(ground_truths.flatten(), predictions.flatten())[0]
        # plt.title("Correlation: " + str(corr))
        # plt.savefig(output_path + "/Prediction_vs_Ground_Truth_(train).png")
        # plt.close()
        
        # get average attention across 
        avg_attention = np.mean(attentions, axis=0)   # neuron_num * neuron_num
        W = avg_attention
        

    elif model_type == "Baseline_2":
        predict_mode_model = baselines.Baseline_2(
            neuron_num=neuron_num,
            learning_rate=learning_rate,
            simulated_network_type=1,
            model_random_seed=model_random_seed,
            loss_function=loss_function,
            log_input=log_input,
        )
        model_checkpoint_path = checkpoint_path + "/" + listdir(checkpoint_path)[-1]

        model_checkpoint = predict_mode_model.load_from_checkpoint(model_checkpoint_path)
        model_checkpoint.eval()
        W = model_checkpoint.W.weight.data
        W = W.cpu().detach().numpy()

        train_results = trainer.predict(predict_mode_model, dataloaders=[trainloader], ckpt_path=model_checkpoint_path)
        # concat train results
        train_results = torch.cat(train_results, dim=0).cpu().detach().numpy()
        predictions = train_results[:, :neuron_num]
        ground_truths = train_results[:, neuron_num:]
        # plot scatter plot of prediction vs. ground truth (flatten)
        corr = stats.pearsonr(ground_truths.flatten(), predictions.flatten())[0]
        plt.scatter(ground_truths.flatten(), predictions.flatten(), s=0.1)
        plt.title("Correlation: " + str(corr))
        plt.xlabel("Ground Truth")
        plt.ylabel("Prediction")
        plt.savefig(output_path + "/Prediction_vs_Ground_Truth_(train).png")
        plt.close()

    

    print("W shape: ", W.shape)
    neuron_types_result = []
    for i in range(len(neuron_types)):
        # split by "-"
        neuron_types_result.append(neuron_types[i].split("-")[0])

    connectivity_matrix_new, connectivity_matrix_cell_type_level, cell_type2cell_type_index = tools.group_connectivity_matrix_by_cell_type(W, neuron_types_result)
    print('cell_type2cell_type_index: ', cell_type2cell_type_index)


    # save onnectivity_matrix_cell_type_level into npy
    np.save(output_path + "/Estimated_W_cell_type_level.npy", connectivity_matrix_cell_type_level)


    plt.imshow(connectivity_matrix_cell_type_level, interpolation='nearest')
    cell_type_labels = cell_type2cell_type_index.keys()
    plt.xticks(np.arange(len(cell_type_labels)), cell_type_labels)
    plt.yticks(np.arange(len(cell_type_labels)), cell_type_labels)
    plt.xlabel("Presynaptic")
    plt.ylabel("Postsynaptic")
    plt.title("Estimated W (cell type level)")
    plt.colorbar()
    plt.show()
    plt.savefig(output_path + "/Estimated_W_cell_type_level.png")
    plt.close()

    # abs
    plt.imshow(np.abs(connectivity_matrix_cell_type_level), interpolation='nearest')
    plt.xticks(np.arange(len(cell_type_labels)), cell_type_labels)
    plt.yticks(np.arange(len(cell_type_labels)), cell_type_labels)
    plt.xlabel("Presynaptic")
    plt.ylabel("Postsynaptic")
    plt.title("abs Estimated W (cell type level)")
    plt.colorbar()
    plt.show()
    plt.savefig(output_path + "/Estimated_W_cell_type_level_abs.png")
    plt.close()

    # negate
    plt.imshow(-connectivity_matrix_cell_type_level, interpolation='nearest')
    plt.xticks(np.arange(len(cell_type_labels)), cell_type_labels)
    plt.yticks(np.arange(len(cell_type_labels)), cell_type_labels)
    plt.xlabel("Presynaptic cell type")
    plt.ylabel("Postsynaptic cell type")
    plt.title("negate Estimated W (cell type level)")
    plt.colorbar()
    plt.show()
    plt.savefig(output_path + "/Estimated_W_cell_type_level_negate.png")
    plt.close()

    # plot
    plt.imshow(connectivity_matrix_new, cmap='bone', interpolation='nearest')
    plt.colorbar()
    # plt.plot((0, 0), (0, 291), c='red', linewidth=5)
    # plt.plot((0, 291), (0, 0), c='red', linewidth=5)

    # plt.plot((0, 0), (291, 291+84), c='blue', linewidth=5)
    # plt.plot((291, 291+84), (0, 0), c='blue', linewidth=5)

    # plt.plot((0, 0), (291+84, 291+84+37), c='green', linewidth=5)
    # plt.plot((291+84, 291+84+37), (0, 0), c='green', linewidth=5)

    # plt.plot((0, 0), (291+84+37, 291+84+37+16), c='yellow', linewidth=5)
    # plt.plot((291+84+37, 291+84+37+16), (0, 0), c='yellow', linewidth=5)

    # plt.plot((0, 0), (291+84+37+16, 291+84+37+16+25), c='purple', linewidth=5)
    # plt.plot((291+84+37+16, 291+84+37+16+25), (0, 0), c='purple', linewidth=5)

    # plt.plot((0, 0), (291+84+37+16+25, 291+84+37+16+25+2), c='orange', linewidth=5)
    # plt.plot((291+84+37+16+25, 291+84+37+16+25+2), (0, 0), c='orange', linewidth=5)

    # plt.plot((0, 0), (291+84+37+16+25+2, 291+84+37+16+25+2+4), c='black', linewidth=5)
    # plt.plot((291+84+37+16+25+2, 291+84+37+16+25+2+4), (0, 0), c='black', linewidth=5)

    plt.xlabel(str(cell_type_labels))
    plt.savefig(output_path + "/Estimated_W.png")
    plt.close()