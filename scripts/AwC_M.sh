python train_model_AwC_M.py --out_folder='../output/4_4_AwC_M/' --model_type=Attention_With_Constraint --window_size=200 --predict_window_size=1 --normalization=session --model_random_seed=42 --dim_key=200 --hidden_size_2=128 --h_layers_2=0 --learning_rate=1e-3 --attention_activation=none --scheduler=plateau --constraint_loss_weight=1e-5 --constraint_var=1 > out2.out


# Add causal temporal map
python train_model_AwC_M.py --out_folder='../output/4_7_AwC_M_softmaxW_out/' --model_type=Attention_With_Constraint --window_size=200 --predict_window_size=1 --normalization=session --model_random_seed=42 --dim_key=200 --hidden_size_2=128 --h_layers_2=0 --learning_rate=1e-3 --attention_activation=none --scheduler=plateau --constraint_loss_weight=0 --constraint_var=1 --causal_temporal_map=lower_triangle --causal_temporal_map_diff=1 --l1_on_causal_temporal_map=5e-5 > out2.out



python train_model_AwC_M.py --out_folder='../output/4_16_AwC_M_noOut_Animal4/' --model_type=Attention_With_Constraint --window_size=200 --predict_window_size=1 --normalization=session --model_random_seed=62 --dim_key=200 --hidden_size_2=128 --h_layers_2=0 --learning_rate=1e-3 --attention_activation=none --scheduler=plateau --constraint_loss_weight=1e-5 --constraint_var=1 --causal_temporal_map=lower_triangle --causal_temporal_map_diff=1 --l1_on_causal_temporal_map=5e-5 > out.out


# add param to change input sessions, make constraint_var learnable
python train_model_AwC_M.py --out_folder='../output/4_17_AwC_M_scaling/' --model_type=Attention_With_Constraint --input_sessions=SB025/2019-10-04/_SB025/2019-10-07/ --window_size=200 --predict_window_size=1 --normalization=session --model_random_seed=42 --dim_key=200 --hidden_size_2=128 --h_layers_2=0 --learning_rate=1e-3 --attention_activation=none --scheduler=plateau --constraint_loss_weight=1e-5 --constraint_var=1 --causal_temporal_map=none --causal_temporal_map_diff=1 --l1_on_causal_temporal_map=0 > out2.out


python train_model_AwC_M.py --out_folder='../output/4_20_AwC_M_SB025_normAll/' --model_type=Attention_With_Constraint --input_mouse=SB025 --input_sessions=2019-10-08_2019-10-23_2019-10-09_2019-10-24_2019-10-07 --window_size=200 --predict_window_size=1 --normalization=session --dim_key=200 --hidden_size_2=128 --h_layers_2=0 --learning_rate=1e-3 --attention_activation=none --scheduler=plateau --constraint_loss_weight=1e-3 --constraint_var=1 --causal_temporal_map=none --causal_temporal_map_diff=1 --l1_on_causal_temporal_map=0 --model_random_seed=42 > out.out