python train_model_Mouse.py --out_folder='../output/10_16_Attention_Mouse/' --model_type=Attention_Autoencoder --data_type=mouse --task_type=prediction --window_size=20 --predict_window_size=1 --heads=1 --attention_layers=1 --hidden_size_2=64 --h_layers_2=0 --learning_rate=5e-4 --pos_enc_type=lookup_table --model_random_seed=42 --dim_key=10 > out.out


# mask task
python train_model_Mouse.py --out_folder='../output/10_18_Attention_Mouse_mask/' --model_type=Attention_Autoencoder --data_type=mouse --task_type=mask --window_size=20 --mask_size=5 --heads=1 --attention_layers=1 --hidden_size_2=64 --h_layers_2=0 --learning_rate=5e-4 --pos_enc_type=lookup_table --model_random_seed=42 --dim_key=10 > out.out


# normalization=session, softmax
python train_model_Mouse.py --out_folder='../output/10_20_Attention_Mouse_mask_softmax/' --model_type=Attention_Autoencoder --data_type=mouse --normalization=session --task_type=mask --window_size=20 --mask_size=5 --heads=1 --attention_layers=1 --hidden_size_2=64 --h_layers_2=0 --learning_rate=5e-4 --pos_enc_type=lookup_table --model_random_seed=42 --dim_key=10 > out.out


# randomly mask elements, normalization=session, softmax
python train_model_Mouse.py --out_folder='../output/10_22_Attention_Mouse_mask_elements/' --model_type=Attention_Autoencoder --data_type=mouse --normalization=session --task_type=mask --window_size=20 --mask_size=100 --heads=1 --attention_layers=1 --hidden_size_2=64 --h_layers_2=0 --learning_rate=1e-3 --pos_enc_type=lookup_table --model_random_seed=42 --dim_key=20 > out.out


# log data, poission likelihood loss
python train_model_Mouse.py --out_folder='../output/10_29_Attention_Mouse_predict_log_poisson_sigmoid/' --model_type=Attention_Autoencoder --data_type=mouse --normalization=log --task_type=prediction --window_size=20 --predict_window_size=1 --heads=1 --attention_layers=1 --hidden_size_2=64 --h_layers_2=0 --learning_rate=1e-3 --pos_enc_type=lookup_table --model_random_seed=22 --dim_key=20 --loss_function=poisson > out.out


python train_model_Mouse.py --out_folder='../output/10_29_Attention_Mouse_predict_log_poisson_sigmoid/' --model_type=Attention_Autoencoder --data_type=mouse --normalization=log --task_type=mask --window_size=20 --mask_size=100 --heads=1 --attention_layers=1 --hidden_size_2=64 --h_layers_2=0 --learning_rate=1e-3 --pos_enc_type=lookup_table --model_random_seed=42 --dim_key=20 --loss_function=poisson --attention_activation=tanh > out.out

