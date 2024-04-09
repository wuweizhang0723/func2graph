python train_model_AwC_M.py --out_folder='../output/4_4_AwC_M/' --model_type=Attention_With_Constraint --window_size=200 --predict_window_size=1 --normalization=session --model_random_seed=42 --dim_key=200 --to_q_layers=0 --to_k_layers=0 --hidden_size_2=128 --h_layers_2=0 --learning_rate=1e-3 --attention_activation=none --scheduler=plateau --constraint_loss_weight=0.00001 --constraint_var=1 > out2.out


# Add causal temporal map
python train_model_AwC_M.py --out_folder='../output/4_7_AwC_M_softmaxW_out/' --model_type=Attention_With_Constraint --window_size=200 --predict_window_size=1 --normalization=session --model_random_seed=42 --dim_key=200 --to_q_layers=0 --to_k_layers=0 --hidden_size_2=128 --h_layers_2=0 --learning_rate=1e-3 --attention_activation=none --scheduler=plateau --constraint_loss_weight=0 --constraint_var=1 --causal_temporal_map=lower_triangle --causal_temporal_map_diff=1 --l1_on_causal_temporal_map=5e-5 > out2.out



python train_model_AwC_M.py --out_folder='../output/4_7_AwC_M_wout/' --model_type=Attention_With_Constraint --window_size=200 --predict_window_size=1 --normalization=session --model_random_seed=12 --dim_key=200 --to_q_layers=0 --to_k_layers=0 --hidden_size_2=128 --h_layers_2=0 --learning_rate=1e-3 --attention_activation=none --scheduler=plateau --constraint_loss_weight=0 --constraint_var=1 --causal_temporal_map=none --causal_temporal_map_diff=1 --l1_on_causal_temporal_map=0 > out4.out