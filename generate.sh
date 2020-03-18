
#! /bin/bash
#
python generate.py --ckpt_path vae_model3/vae-epoch250 \
                   --output_save_path nmd \
                   --n_generations 200 \
                   --style 1 \
                   --output_type all \
                   --temperature 0.2 \
                   --cont_dim 120 \
                   --cat_dim 2

