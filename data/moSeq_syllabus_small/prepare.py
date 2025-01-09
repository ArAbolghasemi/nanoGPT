"""
preapre the moSeq syllabus data to be used by nanoGPT
It will load the nyp data from the dorection in the input_file_path.txt
Then it will consider the unique tokens for each syllabus and create a mapping from characters to integers
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.
"""
import os
import pickle
import requests
import numpy as np

# load the moSeq syllabus data
# read the data from the input_file_path

input_file_path = os.path.join(os.path.dirname(__file__), 'input_file_path.txt')
with open(input_file_path, 'r') as f:
    data_path = f.read().strip()
print(f"loading data from: {data_path}")

# load the data
data = np.load(data_path, allow_pickle=True)
print(f"length of dataset in syllabus: {len(data):,}")

# data are svaed in an array and format of each element is a float32
# get all the unique syllabus that occur in this array of syllabus
syllabus = sorted(list(set(data)))
vocab_size = len(syllabus)
# let us print all the unique syllabus
print("all the unique syllabus:", ''.join(str(syllabus)))
print(f"vocab size: {vocab_size:,}")

# create a mapping from syllabus to integers
stoi = { ch:i for i,ch in enumerate(syllabus) }
itos = { i:ch for i,ch in enumerate(syllabus) }
def encode(s): # encoder: take a np.array, output a list of integers
    return [stoi[c] for c in s]
def decode(l): # decoder: take a list of integers, output a np.array
    out = np.array([itos[i] for i in l])
    return out

# create the train and test splits
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]
# save the val_data to a file for fearther analysis
np.save(os.path.join(os.path.dirname(__file__), 'val_data.npy'), val_data) # TODO: remove this line if the data is large
# save the first 200 elements of the val_data to a file for sample generation
np.save(os.path.join(os.path.dirname(__file__), 'val_data_sample.npy'), val_data[:200])

# encode both to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")
 
# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)


"""
length of dataset in syllabus: 1,080,108
all the unique syllabus: [3.0, 4.0, 10.0, 11.0, 13.0, 14.0, 15.0, 19.0, 20.0, 24.0, 25.0, 30.0, 31.0,
  32.0, 33.0, 35.0, 36.0, 42.0, 43.0, 46.0, 47.0, 48.0, 49.0, 50.0, 51.0, 52.0, 54.0, 55.0, 57.0, 58.0,
  59.0, 60.0, 63.0, 65.0, 66.0, 67.0]
vocab size: 36
train has 972,097 tokens
val has 108,011 tokens
"""
    