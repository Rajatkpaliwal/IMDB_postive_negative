import h5py

with h5py.File("simple_rnn_imdb.h5", "r") as f:
    print(f.keys())  # Should list dataset keys
