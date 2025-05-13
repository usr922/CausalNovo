CUDA_VISIBLE_DEVICES=0 \
python tests/causal_denovo.py \
--mode train \
--data_path ./nine_species | ./seven_species | ./hc_pt \
--config_path configs/config.yaml
