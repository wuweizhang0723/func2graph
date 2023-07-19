import numpy as np
import torch
import pandas as pd
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl



# Implement two ways to look for weight matrix after training is done
# 1) take average of all N attention outputs as weight matrix
# 2) uses sliding windows so that we can visualize if the attention output is smoothly and continuously changed


def get_avg_attention(dataloader, predict_mode_model, checkpoint_path, neuron_num=10):
    trainer = pl.Trainer(
        devices=[1],
        accelerator="gpu",
        benchmark=False,
        profiler="simple",
    )

    # attentions: batch_num * batch_size * neuron_num * neuron_num
    predictions, attentions = trainer.predict(predict_mode_model, dataloaders=[dataloader], ckpt_path=checkpoint_path)
    predictions = torch.cat(predictions, dim=0).cpu().numpy()  # N * neuron_num * window_size
    attentions = torch.cat(attentions, dim=0).cpu().numpy()    # N * neuron_num * neuron_num
    
    # get average attention across 
    attentions = np.mean(attentions, axis=0)   # neuron_num * neuron_num
    return predictions, attentions