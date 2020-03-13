
#! /bin/bash
#
python generate.py --ckpt_path vae_model/vae-epoch300 \
                   --output_save_path jsb \
                   --n_generations 20 \
                   --style 0 \
                   --output_type wav \
                   --x_depth "89 33 33" \
		   --enc_rnn_dim 512 \
		   --enc_rnn_layer 1 \
		   --enc_hyper_unit 256 \
		   --enc_dropout 0.25 \
		   --enc_rnn hyperlstm \
		   --dec_rnn_dim 512 \
		   --dec_hyper_unit 256 \
		   --dec_dropout 0.2 \
		   --dec_rnn_layer 1 \
		   --dec_rnn hyperlstm \
                   --attention 0 \
		   --cont_dim 120 \
		   --cat_dim 2 \
		   --style_embed_dim 80

