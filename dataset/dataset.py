# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 20:45:47 2019

@author: YQ
"""

import glob
import itertools
import pickle

from dataset_converter import build_dataset
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def create_midi_vector(paths, aug=range(1)):
    songs = []
    skipped = []
    for f in tqdm(paths):
        try:
            s = build_dataset(f, False, aug)
            songs.append(s)
        except Exception as e:
            skipped.append((f, e))
    
    songs = list(itertools.chain.from_iterable(songs))
    
    return songs, skipped

        
        
files = glob.glob("bach/chor[0-9][0-9][0-9].krn.mid")
Xtr, Xte = train_test_split(files, test_size=50, random_state=99)
jsb_train, train_skip = create_midi_vector(Xtr)
jsb_test, test_skip = create_midi_vector(Xte)


Xte = glob.glob("Nottingham/test/*") + glob.glob("Nottingham/valid/*")
Xtr, Xte = train_test_split(Xte, test_size=0.05, random_state=99)
nmd_train, train_skip = create_midi_vector(Xtr)
nmd_test, test_skip = create_midi_vector(Xte)


pickle.dump(jsb_train, open("jsb_train.pkl", "wb"))
pickle.dump(jsb_test, open("jsb_test.pkl", "wb"))
pickle.dump(nmd_train, open("nmd_train.pkl", "wb"))
pickle.dump(nmd_test, open("nmd_test.pkl", "wb"))

