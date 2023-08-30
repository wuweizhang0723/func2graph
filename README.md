# func2graph

Install conda environment
```
conda env create --name func2graph --file environment.yml
```

Install the func2graph package
```
conda activate func2graph
pip install -e .
```

Training
- Run the script below to start training model
- Checkpoint with the lowest val_loss will be stored under ```output/training_result```
```
cd scripts
python train_model.py --out_folder='../output/training_result/' --model_type=Attention_Autoencoder --neuron_num=10 --tau=0.3 --data_type=prediction --predict_window_size=10 --hidden_size_1=128 --h_layers_1=2 --heads=1 --attention_layers=1 --hidden_size_2=64 --h_layers_2=2 --learning_rate=5e-5 --pos_enc_type=none --weight_type=random
```

Visualze training and validation using Tensorboard
```
tensorboard --logdir=./output/training_result/log
```

Evaluation
- Change ```checkpoint_path``` in ```notebook/evaluate_prediction_model.ipynb``` and run the entire notebook
- Average attention learned and prediction performance will be visualized in the notebook
