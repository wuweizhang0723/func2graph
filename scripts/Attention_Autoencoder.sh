python train_model.py --out_folder='../output/9_5_Attention_Autoencoder/' --model_type=Attention_Autoencoder --neuron_num=10 --task_type=prediction --predict_window_size=1 --hidden_size_1=128 --h_layers_1=2 --heads=1 --attention_layers=1 --hidden_size_2=64 --h_layers_2=2 --learning_rate=5e-5 --pos_enc_type=lookup_table --weight_type=random --model_random_seed=42 > out.out