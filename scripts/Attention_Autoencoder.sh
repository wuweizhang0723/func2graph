python train_model.py --out_folder='../output/9_9_no_softmax/' --model_type=Attention_Autoencoder --neuron_num=10 --task_type=prediction --window_size=200 --predict_window_size=1 --hidden_size_1=128 --h_layers_1=2 --heads=1 --attention_layers=1 --hidden_size_2=64 --h_layers_2=2 --learning_rate=5e-5 --pos_enc_type=lookup_table --weight_type=random --model_random_seed=42 > out.out



python train_model.py --out_folder='../output/9_11_input_20_identity_v_tanh_noMLP1/' --model_type=Attention_Autoencoder --neuron_num=200 --task_type=prediction --window_size=20 --predict_window_size=1 --hidden_size_1=16 --h_layers_1=0 --heads=1 --attention_layers=1 --hidden_size_2=8 --h_layers_2=0 --learning_rate=5e-5 --pos_enc_type=lookup_table --weight_type=random --model_random_seed=42 > out.out



python train_model.py --out_folder='../output/10_22_data_b_softmax/' --model_type=Attention_Autoencoder --data_type=wuwei --weight_type=random --neuron_num=10 --task_type=prediction --window_size=10 --predict_window_size=1 --hidden_size_1=32 --h_layers_1=0 --heads=1 --attention_layers=1 --hidden_size_2=16 --h_layers_2=0 --learning_rate=5e-5 --pos_enc_type=lookup_table --model_random_seed=42 --data_random_seed=16 --dim_key=10 > out.out


python train_model.py --out_folder='../output/11_5_cell_type_W_AA/' --model_type=Attention_Autoencoder --data_type=wuwei --weight_type=cell_type --neuron_num=200 --task_type=prediction --window_size=200 --predict_window_size=1 --hidden_size_1=32 --h_layers_1=0 --heads=1 --attention_layers=1 --hidden_size_2=64 --h_layers_2=0 --learning_rate=5e-5 --pos_enc_type=lookup_table --model_random_seed=32 --data_random_seed=42 --dim_key=200 --attention_activation=tanh


python train_model.py --out_folder='../output/11_8_strength/' --model_type=Attention_Autoencoder --data_type=wuwei --weight_type=cell_type --neuron_num=200 --task_type=prediction --window_size=200 --predict_window_size=1 --hidden_size_1=32 --h_layers_1=0 --heads=1 --attention_layers=1 --hidden_size_2=64 --h_layers_2=0 --learning_rate=5e-5 --pos_enc_type=lookup_table --model_random_seed=42 --data_random_seed=42 --dim_key=200 --attention_activation=none --scheduler=cycle --weight_decay=0


python train_model.py --out_folder='../output/1_16_AA_toy/' --model_type=Attention_Autoencoder --data_type=wuwei --weight_type=cell_type --neuron_num=200 --task_type=prediction --window_size=200 --predict_window_size=1 --hidden_size_1=32 --h_layers_1=0 --heads=1 --attention_layers=1 --hidden_size_2=64 --h_layers_2=0 --learning_rate=5e-4 --pos_enc_type=lookup_table --model_random_seed=2 --data_random_seed=42 --dim_key=200 --attention_activation=none --scheduler=cycle --weight_decay=0 --to_q_layers=0 --to_k_layers=0 > out.out


# partial observation
python train_model.py --out_folder='../output/1_30_toy_spatialPartial/' --model_type=Attention_Autoencoder --data_type=wuwei --weight_type=cell_type --neuron_num=200 --spatial_partial_measurement=10 --task_type=prediction --window_size=10 --predict_window_size=1 --hidden_size_1=32 --h_layers_1=0 --heads=1 --attention_layers=1 --hidden_size_2=64 --h_layers_2=0 --learning_rate=5e-4 --pos_enc_type=lookup_table --model_random_seed=42 --data_random_seed=42 --dim_key=10 --attention_activation=none --scheduler=cycle --weight_decay=0 --to_q_layers=0 --to_k_layers=0 > out2.out


# causal temporal map, causal temporal map diff, no relative bias
python train_model.py --out_folder='../output/2_20_causalTemporal/' --model_type=Attention_Autoencoder --data_type=wuwei --weight_type=cell_type --neuron_num=200 --spatial_partial_measurement=200 --task_type=prediction --window_size=200 --predict_window_size=1 --hidden_size_1=32 --h_layers_1=0 --heads=1 --attention_layers=1 --hidden_size_2=64 --h_layers_2=0 --learning_rate=1e-3 --pos_enc_type=lookup_table --model_random_seed=42 --data_random_seed=42 --dim_key=200 --attention_activation=none --scheduler=cycle --weight_decay=0 --to_q_layers=0 --to_k_layers=0 --causal_temporal_map=True --causal_temporal_map_diff=1 > out2.out


# l1 on causal temporal map
python train_model.py --out_folder='../output/2_27_causalTemporal_lowerTri/' --model_type=Attention_Autoencoder --data_type=wuwei --weight_type=cell_type --neuron_num=200 --spatial_partial_measurement=200 --task_type=prediction --window_size=200 --predict_window_size=1 --hidden_size_1=32 --h_layers_1=0 --heads=1 --attention_layers=1 --hidden_size_2=64 --h_layers_2=0 --learning_rate=1e-3 --pos_enc_type=lookup_table --model_random_seed=42 --data_random_seed=42 --dim_key=200 --attention_activation=none --scheduler=cycle --weight_decay=0 --to_q_layers=0 --to_k_layers=0 --causal_temporal_map=lower_triangle --causal_temporal_map_diff=1 --l1_on_causal_temporal_map=5e-5 > out2.out