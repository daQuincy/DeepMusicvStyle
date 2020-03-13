# -*- coding: utf-8 -*-
"""
Created on Mon May 27 10:14:38 2019

@author: YQ
"""

from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import pickle
import numpy as np
import itertools

class load_noteseqs:
    def __init__(self, path, x_depth, batch_size=16, augment=True):
        self.data = [pickle.load(open(p, "rb")) for p in path]
        
        self.notes = [d for d in self.data]
        
        self.labels = []
        for i in range(len(self.data)):
            tmp = len(self.data[i])
            self.labels.append(np.ones([tmp])*i)
        self.labels = self.labels[0] if len(self.labels) == 1 else np.concatenate(self.labels, 0)
        
        self.x_depth = x_depth
        
        self.notes = list(itertools.chain.from_iterable(self.notes))
        self.seq_len = [len(x) for x in self.notes]
        
        self.batch_size = batch_size
        self.augment = augment
        self.total_batches = int(len(self.notes) // self.batch_size)
        
        
    def loader(self):
        Z = list(zip(self.notes, self.seq_len, self.labels))
        np.random.shuffle(Z)
        notes, seq_len, labels = zip(*Z)
        for i in range(self.total_batches):
            tmp_notes = notes[self.batch_size*i:(self.batch_size*i)+self.batch_size]
            tmp_seq_len = seq_len[self.batch_size*i:(self.batch_size*i)+self.batch_size]
            tmp_label = labels[self.batch_size*i:(self.batch_size*i)+self.batch_size]
            if len(tmp_notes) == self.batch_size:
                tmp_notes = pad_sequences(tmp_notes, padding="post", dtype=np.int32, value=-1)
                
                if self.augment:
                    aug = np.random.choice(np.arange(-5, 6))
                    pitch = np.roll(tmp_notes[:, :, :88], aug, axis=-1)
                    tmp_notes = np.concatenate([pitch, tmp_notes[:, :, 88:]], -1)
                
                yield tmp_notes, tmp_seq_len, tmp_label
            else:
                break

    def get_iterator(self):
        ds = tf.data.Dataset.from_generator(self.loader, (tf.float32, tf.int32, tf.int32))
        ds = ds.shuffle(self.batch_size*2)
        
        iterate = ds.make_initializable_iterator()
        note, seq_len, label = iterate.get_next()
        note.set_shape([None, None, sum(self.x_depth)])
        seq_len.set_shape([None])
        label.set_shape([None])
        
        return iterate, note, seq_len, label
    

if __name__ == "__main__":
    """
    For testing purposes
    """
    
    import time
    noteseq = load_noteseqs(["data/jsbvl.pkl", "data/nmdvl.pkl", "data/popvl.pkl"], [89, 33, 33])
    it, note, seq_len, label = noteseq.get_iterator()
    sess = tf.Session()
    sess.run(it.initializer)
    
    data = []
    total = 0.0
    tik = time.time()
    while True:
        try:
            data.append(sess.run([note, seq_len, label]))
            tok = time.time()
            print(tok-tik)
            total += tok-tik
            tik = time.time()
        except tf.errors.OutOfRangeError:
            break
        