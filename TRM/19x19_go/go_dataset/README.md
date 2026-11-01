The shared preprocessing script writes one train, validation, and test split
here. Compressed format uses `train.npz`, `val.npz`, and `test.npz`; the
memory-mapped format uses `train/`, `val/`, and `test/` directories, each
containing `X.npy` and `y.npy`. When both are present, training and evaluation
prefer the memory-mapped split by default. Dataset files are local artifacts
and are excluded from Git.
