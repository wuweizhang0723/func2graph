python train_model_AwC_sim.py --out_folder='../output/3_25_AwC/' --model_type=Attention_With_Constraint_sim --data_type=wuwei --weight_type=cell_type --neuron_num=200 --spatial_partial_measurement=200 --task_type=prediction --window_size=200 --predict_window_size=1 --attention_layers=1 --learning_rate=1e-3 --pos_enc_type=lookup_table --model_random_seed=42 --data_random_seed=42 --attention_activation=none --scheduler=cycle --weight_decay=0 --causal_temporal_map=none --causal_temporal_map_diff=1 --l1_on_causal_temporal_map=5e-5 --constraint_loss_weight=0.001 --constraint_var=0.1 > out2.out

# To show TT matrix in the model can capture temporal association, tau variable is used for generating the simulated data.
python train_model_AwC_sim.py --out_folder='../output/4_21_AwC_sim/' --model_type=Attention_With_Constraint_sim --data_type=wuwei --tau=1 --weight_type=cell_type --neuron_num=200 --spatial_partial_measurement=200 --task_type=prediction --window_size=200 --predict_window_size=1 --attention_layers=1 --learning_rate=1e-3 --pos_enc_type=lookup_table --model_random_seed=42 --data_random_seed=42 --attention_activation=none --scheduler=cycle --weight_decay=0 --constraint_loss_weight=0 --constraint_var=0.1 --causal_temporal_map=none --causal_temporal_map_diff=1 --l1_on_causal_temporal_map=0 > out2.out



# For ziyu's data
python train_model_AwC_sim.py --out_folder='../output/6_14_ziyu/' --model_type=Attention_With_Constraint_sim --data_type=ziyu --neuron_num=200 --spatial_partial_measurement=200 --task_type=prediction --window_size=200 --predict_window_size=1 --attention_layers=1 --learning_rate=1e-3 --pos_enc_type=lookup_table --model_random_seed=42 --data_random_seed=42 --attention_activation=none --scheduler=cycle --constraint_loss_weight=0 --constraint_var=0.1 --causal_temporal_map=none --causal_temporal_map_diff=1 --l1_on_causal_temporal_map=0 > out2.out



# Add out_layer parameter to the model
python train_model_AwC_sim.py --out_folder='../output/5_12_AwC_sim/' --model_type=Attention_With_Constraint_sim --data_type=wuwei --tau=1 --out_layer=False --weight_type=cell_type --neuron_num=200 --spatial_partial_measurement=200 --task_type=prediction --window_size=200 --predict_window_size=1 --learning_rate=1e-3 --pos_enc_type=lookup_table --model_random_seed=42 --data_random_seed=42 --attention_activation=none --scheduler=cycle --weight_decay=0 --constraint_loss_weight=0 --constraint_var=0.1 --causal_temporal_map=none --causal_temporal_map_diff=1 --l1_on_causal_temporal_map=5e-5