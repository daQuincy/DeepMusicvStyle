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
ap.add_argument("--ckpt_path", default="vae_model2/vae-epoch235", type=str)
ap.add_argument("--output_type", default="midi", type=str)
ap.add_argument("--temperature", default=1.0, type=float)
ap.add_argument("--style", default=0, type=int)
ap.add_argument("--output_save_path", type=str)
ap.add_argument("--n_generations", default=20, type=int)
ap.add_argument("--cont_dim", default=120, type=int)
ap.add_argument("--cat_dim", default=2, type=int)

args = ap.parse_args()

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


saver = tf.train.import_meta_graph(args.ckpt_path+".meta")
temperature, output, length, mz_cont, mz_cat = tf.get_collection("restore_ops")

tf_config = tf.ConfigProto()
tf_config.allow_soft_placement = True
tf_config.gpu_options.allow_growth = True

sess = tf.Session(config=tf_config)
saver.restore(sess, args.ckpt_path)
    
z_cont = np.random.normal(size=(args.n_generations, args.cont_dim))
z_cat = np.zeros((args.n_generations, args.cat_dim), np.float32)
z_cat[:, args.style] = 1.

fd = {mz_cont: z_cont, mz_cat: z_cat, temperature: args.temperature}
outputs, lengths = sess.run([output, length], feed_dict=fd)
for i, output in enumerate(outputs):
    p, d, r = np.split(output, 3, axis=-1)
    mid = vec2midi(p, d, r)
    
    if args.output_type == "midi":
        mid.write("{}/{}.mid".format(args.output_save_path, i))
    elif args.output_type == "wav":
        audio = mid.fluidsynth(44100, "steinway.sf2").astype(np.float32)
        wavfile.write("{}/{}.wav".format(args.output_save_path, i), 44100, audio)
    elif args.output_type == "all":
        mid.write("{}/{}.mid".format(args.output_save_path, i))
        audio = mid.fluidsynth(44100, "steinway.sf2").astype(np.float32)
        wavfile.write("{}/{}.wav".format(args.output_save_path, i), 44100, audio)



