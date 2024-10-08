############################################
# 1. Scaling
############################################

# SB025 Single Session:
# seed 12-22
# no prior
# window_size=60

# SB025 2019-10-23:
python train_model_AwC_M.py --out_folder='../output/5_13_oneSession/' --model_type=Attention_With_Constraint --input_mouse='SB025' --input_sessions='2019-10-23' --window_size=60 --predict_window_size=1 --normalization=session --learning_rate=1e-3 --attention_activation=none --scheduler=plateau --constraint_loss_weight=0 --constraint_var=1 --causal_temporal_map=none --causal_temporal_map_diff=1 --l1_on_causal_temporal_map=0 --model_random_seed=12
# SB025 2019-10-24:
python train_model_AwC_M.py --out_folder='../output/5_13_oneSession/' --model_type=Attention_With_Constraint --input_mouse='SB025' --input_sessions='2019-10-24' --window_size=60 --predict_window_size=1 --normalization=session --learning_rate=1e-3 --attention_activation=none --scheduler=plateau --constraint_loss_weight=0 --constraint_var=1 --causal_temporal_map=none --causal_temporal_map_diff=1 --l1_on_causal_temporal_map=0 --model_random_seed=12

# SB025: 
# seed 12-52
# no prior
# window_size=60
python train_model_AwC_M.py --out_folder='../output/5_13_SB025/' --model_type=Attention_With_Constraint --input_mouse='SB025' --input_sessions='2019-10-04_2019-10-07_2019-10-08_2019-10-09_2019-10-23_2019-10-24' --window_size=60 --predict_window_size=1 --normalization=session --learning_rate=1e-3 --attention_activation=none --scheduler=plateau --constraint_loss_weight=0 --constraint_var=1 --causal_temporal_map=none --causal_temporal_map_diff=1 --l1_on_causal_temporal_map=0 --model_random_seed=12

# SB025 + SB028:
# seed 12-52
# no prior
# window_size=60
python train_model_AwC_M.py --out_folder='../output/5_13_SB025_SB028/' --model_type=Attention_With_Constraint --input_mouse='SB025|SB028' --input_sessions='2019-10-04_2019-10-07_2019-10-08_2019-10-09_2019-10-23_2019-10-24|2019-11-06_2019-11-07_2019-11-08_2019-11-12_2019-11-13' --window_size=60 --predict_window_size=1 --normalization=session --learning_rate=1e-3 --attention_activation=none --scheduler=plateau --constraint_loss_weight=0 --constraint_var=1 --causal_temporal_map=none --causal_temporal_map_diff=1 --l1_on_causal_temporal_map=0 --model_random_seed=12



############################################
# 2. Prior
############################################

# SB025 2019-10-24:
# seed 12-102
# no prior
# window_size=60

# SB025 2019-10-24:
# seed 12-102
# with prior
# window_size=60

# SB025:
# seed 12-102
# no prior
# window_size=60

# SB025:
# seed 12-102
# with prior
# window_size=60




############################################
# 3. Baselines
############################################

###################################### Real Data SB025 2019-10-23
# 1. GLM_M w/ tanh
# seed 42
# normalization=all
python train_model_GLM_M.py --out_folder='../output/5_16_GLM_M/' --model_type=GLM_M --k=1 --input_mouse='SB025' --input_sessions='2019-10-23' --batch_size=32 --normalization=all --model_random_seed=42 --learning_rate=1e-4 --scheduler=plateau --weight_decay=0 --activation_type=tanh
# 2. GLM_M w/ exp
# seed 42
# normalization=destd
python train_model_GLM_M.py --out_folder='../output/5_16_GLM_M/' --model_type=GLM_M --k=1 --input_mouse='SB025' --input_sessions='2019-10-23' --batch_size=32 --normalization=destd --model_random_seed=42 --learning_rate=1e-4 --scheduler=plateau --weight_decay=0 --activation_type=exp
# 3. AwC_M
# seed 22
# no prior
# window_size=60
python train_model_AwC_M.py --out_folder='../output/2019-10-23/' --model_type=Attention_With_Constraint --input_mouse='SB025' --input_sessions='2019-10-23' --window_size=60 --predict_window_size=1 --normalization=session --learning_rate=1e-3 --attention_activation=none --scheduler=plateau --constraint_loss_weight=0 --constraint_var=1 --causal_temporal_map=none --causal_temporal_map_diff=1 --l1_on_causal_temporal_map=0 --model_random_seed=22
# 4. AwC_M2
# seed 42
# no prior
# window_size=60, dim_E=30
python train_model_AwC_M.py --out_folder='../output/AwC_M2/' --model_type=Attention_With_Constraint_2 --input_mouse='SB025' --input_sessions='2019-10-23' --window_size=60 --predict_window_size=1 --normalization=session --model_random_seed=42 --learning_rate=1e-3 --scheduler=plateau --constraint_loss_weight=0 --constraint_var=1 --causal_temporal_map=none --causal_temporal_map_diff=1 --l1_on_causal_temporal_map=0 --dim_E=30

###################################### Simulation Data tau=1, (NeuroAI workshop: weight_scale=1, init_scale=1, error_scale=3.5)
# 1. GLM_sim w/ tanh
# seed 42
python train_model_GLM_sim.py --out_folder='../output/5_17_GLM_sim/' --model_type=GLM_sim --data_type=wuwei --weight_scale=1 --init_scale=1 --error_scale=3.5 --tau=1 --weight_type=cell_type --neuron_num=200 --spatial_partial_measurement=200 --predict_window_size=1 --learning_rate=1e-3 --model_random_seed=42 --data_random_seed=42 --scheduler=cycle --weight_decay=0 --activation_type=tanh
# 2. GLM_sim w/ exp
# seed 42
python train_model_GLM_sim.py --out_folder='../output/5_17_GLM_sim/' --model_type=GLM_sim --data_type=wuwei --weight_scale=1 --init_scale=1 --error_scale=3.5 --tau=1 --weight_type=cell_type --neuron_num=200 --spatial_partial_measurement=200 --predict_window_size=1 --learning_rate=1e-3 --model_random_seed=42 --data_random_seed=42 --scheduler=cycle --weight_decay=0 --activation_type=exp
# 3. AwC_sim
# seed 42
# no prior
# window_size=200
python train_model_AwC_sim.py --out_folder='../output/5_15_AwC_sim/' --model_type=Attention_With_Constraint_sim --data_type=wuwei --weight_scale=1 --init_scale=1 --error_scale=3.5 --tau=1 --weight_type=cell_type --neuron_num=200 --spatial_partial_measurement=200 --task_type=prediction --window_size=200 --predict_window_size=1 --learning_rate=1e-3 --model_random_seed=42 --data_random_seed=42 --attention_activation=none --scheduler=cycle --constraint_loss_weight=0 --constraint_var=0.1 --causal_temporal_map=none --causal_temporal_map_diff=1 --l1_on_causal_temporal_map=0
# 4. AwC_sim2
# seed 42
# no prior
# window_size=100, dim_E=200
python train_model_AwC_sim.py --out_folder='../output/8_29_AwC_sim2/' --model_type=Attention_With_Constraint_2_sim --data_type=wuwei --weight_scale=1 --init_scale=1 --error_scale=3.5 --tau=1 --weight_type=cell_type --neuron_num=200 --spatial_partial_measurement=200 --task_type=prediction --window_size=100 --predict_window_size=1 --learning_rate=1e-3 --model_random_seed=12 --data_random_seed=42 --attention_activation=none --scheduler=cycle --constraint_loss_weight=0 --constraint_var=0.1 --causal_temporal_map=none --causal_temporal_map_diff=1 --l1_on_causal_temporal_map=0 --dim_E=200



############################################
# 4. Cell Type Classification
############################################

# AwC_M
# SB025 2019-10-23:
# seed 42
# no prior
# window_size=60
python train_model_AwC_M.py --out_folder='../output/5_13_oneSession/' --model_type=Attention_With_Constraint --input_mouse='SB025' --input_sessions='2019-10-23' --window_size=60 --predict_window_size=1 --normalization=session --learning_rate=1e-3 --attention_activation=none --scheduler=plateau --constraint_loss_weight=0 --constraint_var=1 --causal_temporal_map=none --causal_temporal_map_diff=1 --l1_on_causal_temporal_map=0 --model_random_seed=42

# AwC_M
# SB025: 
# seed 62
# no prior
# window_size=60
python train_model_AwC_M.py --out_folder='../output/5_13_SB025/' --model_type=Attention_With_Constraint --input_mouse='SB025' --input_sessions='2019-10-04_2019-10-07_2019-10-08_2019-10-09_2019-10-23_2019-10-24' --window_size=60 --predict_window_size=1 --normalization=session --learning_rate=1e-3 --attention_activation=none --scheduler=plateau --constraint_loss_weight=0 --constraint_var=1 --causal_temporal_map=none --causal_temporal_map_diff=1 --l1_on_causal_temporal_map=0 --model_random_seed=62


# AwC_sim
# Simulation Data
# tau=1
# seed 42
# no prior
# window_size=200
# causal_temporal_map=none, l1_on_causal_temporal_map=0
python train_model_AwC_sim.py --out_folder='../output/5_12_AwC_sim/' --model_type=Attention_With_Constraint_sim --data_type=wuwei --weight_scale=1 --init_scale=1 --error_scale=3.5 --tau=1 --weight_type=cell_type --neuron_num=200 --spatial_partial_measurement=200 --task_type=prediction --window_size=200 --predict_window_size=1 --learning_rate=1e-3 --pos_enc_type=lookup_table --model_random_seed=42 --data_random_seed=42 --attention_activation=none --scheduler=cycle --weight_decay=0 --constraint_loss_weight=0 --constraint_var=0.1 --causal_temporal_map=none --causal_temporal_map_diff=1 --l1_on_causal_temporal_map=0




############################################
# 5. Temporal Association
############################################

# AwC_M
# SB025 2019-10-23:
# seed 42
# no prior
# window_size=200
# causal_temporal_map=lower_triangle, l1_on_causal_temporal_map=1e-4
python train_model_AwC_M.py --out_folder='../output/5_13_SB025/' --model_type=Attention_With_Constraint --input_mouse='SB025' --input_sessions='2019-10-23' --window_size=200 --predict_window_size=1 --normalization=session --learning_rate=1e-3 --attention_activation=none --scheduler=plateau --constraint_loss_weight=0 --constraint_var=1 --causal_temporal_map=lower_triangle --causal_temporal_map_diff=1 --l1_on_causal_temporal_map=1e-4 --model_random_seed=42


# AwC_sim
# Simulation Data
# tau=10, out_layer=True
# seed 42
# no prior
# window_size=200
# causal_temporal_map=lower_triangle, l1_on_causal_temporal_map=5e-5
python train_model_AwC_sim.py --out_folder='../output/5_13_sim/' --model_type=Attention_With_Constraint_sim --data_type=wuwei --weight_scale=1 --init_scale=1 --error_scale=3.5 --tau=10 --out_layer=True --weight_type=cell_type --neuron_num=200 --spatial_partial_measurement=200 --task_type=prediction --window_size=200 --predict_window_size=1 --learning_rate=1e-3 --pos_enc_type=lookup_table --model_random_seed=42 --data_random_seed=42 --attention_activation=none --scheduler=cycle --weight_decay=0 --constraint_loss_weight=0 --constraint_var=0.1 --causal_temporal_map=lower_triangle --causal_temporal_map_diff=1 --l1_on_causal_temporal_map=5e-5

# AwC_sim
# tau=1, out_layer=False
# seed 42
# no prior
# window_size=200
# causal_temporal_map=lower_triangle, l1_on_causal_temporal_map=5e-5
python train_model_AwC_sim.py --out_folder='../output/5_13_sim/' --model_type=Attention_With_Constraint_sim --data_type=wuwei --weight_scale=1 --init_scale=1 --error_scale=3.5 --tau=1 --out_layer=False --weight_type=cell_type --neuron_num=200 --spatial_partial_measurement=200 --task_type=prediction --window_size=200 --predict_window_size=1 --learning_rate=1e-3 --pos_enc_type=lookup_table --model_random_seed=42 --data_random_seed=42 --attention_activation=none --scheduler=cycle --weight_decay=0 --constraint_loss_weight=0 --constraint_var=0.1 --causal_temporal_map=lower_triangle --causal_temporal_map_diff=1 --l1_on_causal_temporal_map=5e-5





############################################
# 6. Ablation: Activation Function
############################################

# AwC_M
# SB025 2019-10-23: (Softmax)
# seed 22
# no prior
# window_size=60
python train_model_AwC_M.py --out_folder='../output/5_17_AcM_M_activation/' --model_type=Attention_With_Constraint --input_mouse='SB025' --input_sessions='2019-10-23' --window_size=60 --predict_window_size=1 --normalization=session --learning_rate=1e-3 --attention_activation=softmax --scheduler=plateau --constraint_loss_weight=0 --constraint_var=1 --causal_temporal_map=none --causal_temporal_map_diff=1 --l1_on_causal_temporal_map=0 --model_random_seed=22




############################################
# 7. AwC_M2: E_concat model architecture
############################################

# AwC_M2
# SB025 2019-10-23
# seed 42
# no prior
# window_size=60
# dim_E=30
python train_model_AwC_M.py --out_folder='../output/8_2_AwC_M2/' --model_type=Attention_With_Constraint_2 --input_mouse='SB025' --input_sessions='2019-10-23' --window_size=60 --predict_window_size=1 --normalization=session --model_random_seed=42 --learning_rate=1e-3 --scheduler=plateau --constraint_loss_weight=0 --constraint_var=1 --causal_temporal_map=none --causal_temporal_map_diff=1 --l1_on_causal_temporal_map=0 --dim_E=30




############################################
# 7. AwC_sim2: E_concat model architecture
############################################
python train_model_AwC_sim.py --out_folder='../output/8_21_AwC_sim2/' --model_type=Attention_With_Constraint_2_sim --data_type=wuwei --weight_scale=1 --init_scale=1 --error_scale=3.5 --tau=1 --weight_type=cell_type --neuron_num=200 --spatial_partial_measurement=200 --task_type=prediction --window_size=100 --predict_window_size=1 --learning_rate=1e-3 --model_random_seed=42 --data_random_seed=42 --attention_activation=none --scheduler=cycle --constraint_loss_weight=0 --constraint_var=0.1 --causal_temporal_map=none --causal_temporal_map_diff=1 --l1_on_causal_temporal_map=0 --dim_E=200





############################################
# Appendix 1. Window Size
############################################

# AwC_M
# SB025 2019-10-23:
# seed 22
# no prior
# window_size=10, 30, 50, 60, 80, 100
python train_model_AwC_M.py --out_folder='../output/5_19_AcM_M_window/' --model_type=Attention_With_Constraint --input_mouse='SB025' --input_sessions='2019-10-23' --window_size=10 --predict_window_size=1 --normalization=session --learning_rate=1e-3 --attention_activation=none --scheduler=plateau --constraint_loss_weight=0 --constraint_var=1 --causal_temporal_map=none --causal_temporal_map_diff=1 --l1_on_causal_temporal_map=0 --model_random_seed=22

# AwC_sim
# tau=1
# seed 42
# no prior
# window_size=10, 50, 100, 150, 200, 250, 300
python train_model_AwC_sim.py --out_folder='../output/5_19_AwC_sim_window/' --model_type=AwC_sim --data_type=wuwei --weight_scale=1 --init_scale=1 --error_scale=3.5 --tau=1 --weight_type=cell_type --neuron_num=200 --spatial_partial_measurement=200 --task_type=prediction --window_size=10 --predict_window_size=1 --learning_rate=1e-3 --model_random_seed=42 --data_random_seed=42 --attention_activation=none --scheduler=cycle --constraint_loss_weight=0 --constraint_var=0.1 --causal_temporal_map=none --causal_temporal_map_diff=1 --l1_on_causal_temporal_map=0




############################################
# Ziyu's Izhikevich Model
############################################

python train_model_AwC_sim.py --out_folder='../output/6_18_ziyu/' --model_type=AwC_sim --data_type=ziyu --neuron_num=200 --spatial_partial_measurement=200 --task_type=prediction --window_size=200 --predict_window_size=1 --attention_layers=1 --learning_rate=1e-3 --pos_enc_type=lookup_table --model_random_seed=42 --data_random_seed=42 --attention_activation=none --scheduler=cycle --constraint_loss_weight=0 --constraint_var=0.1 --causal_temporal_map=none --causal_temporal_map_diff=1 --l1_on_causal_temporal_map=0