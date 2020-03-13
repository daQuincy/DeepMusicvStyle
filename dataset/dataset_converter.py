import pretty_midi as pm
import numpy as np

from collections import Counter

class OneHotEncoder:
    def __init__(self, depth, axis=-1):
        self.depth = depth
        self.axis = axis
        
    def _onehot(self, data):
        oh = np.zeros((self.depth), dtype=np.uint8)
        if data >= 0 and data < self.depth:
            data = int(data)
            oh[data] = 1
        
        return oh
    
    def transform(self, data_list):
        one_hot_encoded = [self._onehot(data) for data in data_list]
        one_hot_encoded = np.stack(one_hot_encoded, axis=0)
        
        return one_hot_encoded
    

def build_dataset(song_path, include_velocity=True, augment=range(1)):
    mid = pm.PrettyMIDI(song_path)
        
    
    # get time signature
    numerators = [t.numerator for t in mid.time_signature_changes]
    denominators = [t.denominator for t in mid.time_signature_changes]
    count = Counter(numerators)
    numerator = sorted(numerators, key=lambda x: count[x], reverse=True)[0]
    count = Counter(denominators)
    denominator = sorted(denominators, key=lambda x: count[x], reverse=True)[0]    

    # extract all notes from non-drum instruments
    midi_note = []
    for ins in mid.instruments:
        if not ins.is_drum:
            for n in ins.notes:
                midi_note.append((n.pitch, n.start, n.end, n.velocity))

    midi_note = sorted(midi_note, key=lambda x: (x[1], x[0]))

    # create features [pitch, velocity, dt, duration, start_time]
    prev_start = 0
    song = []
    for m in midi_note:

        t = mid.time_to_tick(m[1]) - prev_start
        
        pitch = m[0]
        song.append((np.clip(pitch, 21, 108), m[3], t/mid.resolution, mid.time_to_tick(m[2]-m[1])/mid.resolution, m[1]))

        prev_start = mid.time_to_tick(m[1])


    # create list of non-overlapping segments of 4 bars
    time_per_bar = mid.tick_to_time(numerator * mid.resolution * (4/denominator))
    total_bars = int((mid.get_end_time()//time_per_bar))
    bars = []

    tmp = []
    for i in range(0, total_bars, 4):
        for m in song:
            if m[-1] >= (i*time_per_bar) and m[-1] < ((i*time_per_bar) + (time_per_bar*4)):
                tmp.append(m[:-1])
        
        bars.append(tmp)
        tmp = []
    
    # keep only segments that have more than 5 note events
    bars = [np.stack(b) for b in bars if len(b)>= 5]
    
    p_ohe = OneHotEncoder(89)
    t_ohe = OneHotEncoder(33)
    
    
    X = []
    for bb in bars:
        b = np.split(bb, 4, -1)
        b = [np.squeeze(i) for i in b]
        
            
        for i in augment:
            P, V, D, R = [], [], [], []
            p = p_ohe.transform(b[0]-21+i)
            
            d = np.minimum(np.round((b[2]/4) * 32), 32)
            d = t_ohe.transform(d)
            
            r = np.minimum(np.round((b[3]/4) * 32), 32)
            r = t_ohe.transform(r)
            
            v = (b[1] / 63.5) - 1
            v = v.astype(np.float32)
            
            P.append(p)
            V.append(v)
            D.append(d)
            R.append(r)
            
            
            P = np.concatenate(P, axis=0)
            V = np.expand_dims(np.concatenate(V, axis=0), -1)
            D = np.concatenate(D, axis=0)
            R = np.concatenate(R, axis=0)
            
            if include_velocity:
                tmp = np.concatenate([P, D, R, V], -1)
                END_TOKEN = np.zeros(dtype=np.float32, shape=(1, tmp.shape[-1]))
                END_TOKEN[0, 88] = 1.0
                tmp = np.concatenate([tmp, END_TOKEN], 0)
            else:
                tmp = np.concatenate([P, D, R], -1)
                END_TOKEN = np.zeros(dtype=np.float32, shape=(1, tmp.shape[-1]))
                END_TOKEN[0, 88] = 1.0
                tmp = np.concatenate([tmp, END_TOKEN], 0)
                tmp = tmp.astype(np.uint8)
                   
            X.append(tmp)
            
    return X