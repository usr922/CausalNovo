CUDA_VISIBLE_DEVICES=0 \
python tests/causal_denovo.py \
--mode denovo \
--data_path ./nine_species | ./seven_species | ./hc_pt \
--ckpt_path ./saved_models/saved_model.ckpt \
--denovo_output_path results/results.csv \
--config_path configs/config.yaml
