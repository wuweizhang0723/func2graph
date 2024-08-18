import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches 
import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.manifold import TSNE
from scipy import stats
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression

from func2graph import models, data, tools


####################################################################################################
# Real Mouse Data
####################################################################################################

window_size = 60   #############
batch_size = 32

train_dataloader, val_dataloader, num_unqiue_neurons, cell_type_order, all_sessions_new_cell_type_id, num_batch_per_session_TRAIN, num_batch_per_session_VAL, sessions_2_original_cell_type, neuron_id_2_cell_type_id = data.generate_mouse_all_sessions_data(
    input_mouse=['SB025'],              ##########################
    input_sessions=[['2019-10-23']],    ####### '2019-10-04', '2019-10-07', '2019-10-08', '2019-10-09', '2019-10-23', '2019-10-24'
    window_size=window_size,
    batch_size=batch_size,
)

checkpoint_path = "../../output/8_2_AwC_M2_layerNorm_x_e/Attention_With_Constraint_2_SB025_2019-10-23_32_60_1_session_42_128_2_1_1_64_258_0_0.2_0.001_mse_softmax_plateau_0.0_0.0_1.0_none_1_0.0_30/epoch=99-step=9600.ckpt"   #############
trained_model = models.Attention_With_Constraint_2.load_from_checkpoint(checkpoint_path)
trained_model.eval()

indices = torch.tensor([0,1,2]).cuda()
neuron_embeddings = trained_model.embedding_table(indices)
print('neuron_embeddings:', neuron_embeddings.shape)

print('EC:', len(neuron_id_2_cell_type_id[neuron_id_2_cell_type_id==0]))
print('IN:', len(neuron_id_2_cell_type_id[neuron_id_2_cell_type_id!=0]))
print('\n')

for i in range(len(cell_type_order)):
    print(cell_type_order[i], len(neuron_id_2_cell_type_id[neuron_id_2_cell_type_id==i]))


##########################
# Binary Classification
##########################

indices = torch.arange(num_unqiue_neurons).cuda()
neuron_embeddings = trained_model.embedding_table(indices)

all_X = neuron_embeddings.detach().cpu().numpy()
all_y = neuron_id_2_cell_type_id.copy()

# randomly split the data into train and test

np.random.seed(52)  # E_add: 72, 122, E_concat: 52, 172
train_indices = np.random.choice(len(all_y), int(len(all_y) * 0.75), replace=False)
test_indices = np.setdiff1d(np.arange(len(all_y)), train_indices)
X_train = all_X[train_indices]
y_train = all_y[train_indices]
X_test = all_X[test_indices]
y_test = all_y[test_indices]

print('original X_train:', X_train.shape)
print('original X_test:', X_test.shape)

# Make EC and IN balanced in train set

X_train_EC = X_train[y_train==0]
y_train_EC = y_train[y_train==0]
X_train_IN = X_train[y_train!=0]
y_train_IN = y_train[y_train!=0]

X_train_EC = X_train_EC[:len(X_train_IN)]
y_train_EC = y_train_EC[:len(y_train_IN)]

X_train = np.concatenate((X_train_EC, X_train_IN), axis=0)
y_train = np.concatenate((y_train_EC, y_train_IN), axis=0)

# Shuffle the train set

indices = np.arange(len(y_train))
np.random.shuffle(indices)
X_train = X_train[indices]
y_train = y_train[indices]

print('X_train EC:', X_train_EC.shape)
print('X_train IN:', X_train_IN.shape)
print('X_test EC:', X_test[y_test==0].shape)
print('X_test IN:', X_test[y_test!=0].shape)

# For binary classification, make EC as 0 and IN as 1

y_train[y_train!=0] = 1
y_test[y_test!=0] = 1


# Train a logistic regression model

np.random.seed(42)

clf = LogisticRegression(random_state=42).fit(X_train, y_train)
print(clf.score(X_train, y_train))
print('EC IN proportion in TRAIN', np.unique(y_train, return_counts=True)[1] / len(y_train))

print(clf.score(X_test, y_test))
print('EC IN proportion in TEST', np.unique(y_test, return_counts=True)[1] / len(y_test))

# compute auroc
y_pred = clf.predict_proba(X_test)[:, 1]
print('auroc', roc_auc_score(y_test, y_pred))

# get confusion matrix
y_pred = clf.predict(X_test)
# convert confusion matrix to percentage
cm = confusion_matrix(y_test, y_pred)
cm = cm / cm.sum(axis=1)[:, np.newaxis]
plt.imshow(cm, cmap='Blues', interpolation='nearest')
plt.colorbar()
# label percentage on the plot
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, f"{cm[i, j]:.2f}", ha='center', va='center', color='white' if cm[i, j] > thresh else 'black')

# label the axes
plt.xticks([0, 1], ['EC', 'IN'])
plt.yticks([0, 1], ['EC', 'IN'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.clim(0, 1)
plt.savefig('confusion_matrix.png')
plt.close()