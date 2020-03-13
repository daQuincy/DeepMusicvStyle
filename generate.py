# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 01:06:06 2019

@author: YQ
"""

from model import MusicVAE
from scipy.io import wavfile

import pretty_midi
import numpy as np
import tensorflow as tf
import argparse

tf.reset_default_graph()

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt_path", default="../ckpt/vq_35/vae-epoch140", type=str)
ap.add_argument("--output_type", default="midi", type=str)

ap.add_argument("--x_depth", default="89 33 33", type=str)

ap.add_argument("--enc_rnn", default="hyperlstm", type=str)
ap.add_argument("--enc_rnn_dim", default=512, type=int)
ap.add_argument("--enc_hyper_unit", default=256, type=int)
ap.add_argument("--enc_dropout", default=0.0, type=float)
ap.add_argument("--enc_rnn_layer", default=1, type=int)

ap.add_argument("--dec_rnn", default="hyperlstm", type=str)
ap.add_argument("--dec_rnn_dim", default=512, type=int)
ap.add_argument("--dec_hyper_unit", default=256, type=int)
ap.add_argument("--dec_dropout", default=0.0, type=float)
ap.add_argument("--dec_rnn_layer", default=1, type=int)

ap.add_argument("--attention", default=0, type=int)

ap.add_argument("--cont_dim", default=50, type=int)
ap.add_argument("--cat_dim", default=2,type=int)
ap.add_argument("--style_embed_dim", default=150, type=int)

ap.add_argument("--style", default=0, type=int)
ap.add_argument("--output_save_path", type=str)
ap.add_argument("--n_generations", type=int)

args = ap.parse_args()
x_depth = args.x_depth.split()
x_depth = [int(i) for i in x_depth]

def vec2midi(pitches, dts, durations, velocities=None, resolution=120, 
             program=0, initial_tempo=100):
    mid = pretty_midi.PrettyMIDI(resolution=resolution, initial_tempo=initial_tempo)
    piano = pretty_midi.Instrument(program=0)

    start = 0
    for i in range(len(pitches)):
        if int(pitches[i]) == 88:
            break
        if i == 0:
            start = 0
        else:
            start += mid.tick_to_time(int(np.round((dts[i]/8)*mid.resolution)))

        pitch = int(pitches[i]+21)
        if velocities is None:
            velocity = 100 
        else:
            velocity = int((velocities[i]+1)*63.5)
        duration = mid.tick_to_time(int(np.round((durations[i]/8)*mid.resolution)))
        end = start + duration
        
        note = pretty_midi.Note(pitch=pitch, velocity=velocity, start=start, end=end)
        piano.notes.append(note)

    mid.instruments.append(piano)    
    
    return mid

model = MusicVAE(x_depth=x_depth,
                 enc_rnn_dim=args.enc_rnn_dim, enc_hyper_unit=args.enc_hyper_unit, enc_dropout=args.enc_dropout,
                 dec_rnn_dim=args.dec_rnn_dim, dec_hyper_unit=args.dec_hyper_unit, dec_dropout=args.dec_dropout,
                 enc_rnn_layer=args.enc_rnn_layer, dec_rnn_layer=args.dec_rnn_layer,
                 enc_rnn=args.enc_rnn, dec_rnn=args.dec_rnn,
                 attention=args.attention,
                 cont_dim=args.cont_dim,
                 style_embed_dim=args.style_embed_dim,
                 training=False)
model.build()

tf_config = tf.ConfigProto()
tf_config.allow_soft_placement = True
tf_config.gpu_options.allow_growth = True

sess = tf.Session(config=tf_config)
model.saver.restore(sess, args.ckpt_path)

z_cat = np.zeros((1, args.cat_dim), np.float32)
z_cat[0, args.style] = 1.
for i in range(args.n_generations):
    z_cont = np.random.normal(size=(1, args.cont_dim))
    tmp = []
                
    fd = {model.z_cont: z_cont, model.z_cat: z_cat, model.temperature: 0.3}
    output, length = sess.run([model.output, model.len], feed_dict=fd)
    tmp.append(length)
    p, d, r = np.split(output[0], 3, axis=-1)

    mid = vec2midi(p, d, r)
    
    if args.output_type == "midi":
        mid.write("{}/{}.mid".format(args.output_save_path, i))
    elif args.output_type == "wav":
        audio = mid.fluidsynth(44100, "salamander.sf2").astype(np.float32)
        wavfile.write("{}/{}.wav".format(args.output_save_path, i), 44100, audio)
