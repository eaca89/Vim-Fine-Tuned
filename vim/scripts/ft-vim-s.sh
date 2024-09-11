#!/bin/bash
conda activate Vim
cd /home/eh_abdol/Vim/vim;

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
--nproc_per_node=8 \
--use_env main.py \
--model vim_small_patch16_stride8_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 \
--batch-size 128 \
--lr 5e-6 \
--min-lr 1e-5 \
--warmup-lr 1e-5 \
--drop-path 0.0 --weight-decay 1e-8 \
--num_workers 25 --data-path <path_to_IN1K_dataset> \
--output_dir ./output/vim_small_patch16_stride8_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 \
--epochs 30 \
--finetune <path_to_pt_ckpt> \
--no_amp

# The ft-vim-s.sh script you found is indeed responsible for fine-tuning a model 
# by passing specific arguments to main.py.

# TO RUN IT
# Modify the placeholders:

# Replace <your_env> with your environment name.
# Replace <path_to_Vim> with your Vim project path.
# Replace <path_to_IN1K_dataset> with the path to your dataset.
# Replace <path_to_pt_ckpt> with the path to the pretrained checkpoint.

# bash ft-vim-s.sh


