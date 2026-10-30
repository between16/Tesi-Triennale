# 19x19 Go data

This directory is reserved for 19x19 Go data. SGF archives can be filtered
into `go_raw_dataset/19x19_games/` by running the existing extraction helper
with `--board-size 19`, or copied there directly. Then build
train/validation/test splits with:

```bash
python 9x9_go/go_raw_dataset/extract_9x9.py --board-size 19 --source ./games
python go_preprocessing.py --board-size 19
```

The generated files are written to `go_dataset/{train,val,test}.npz`. Training
uses the same board-size parameter and discovers this directory automatically:

```bash
python train.py --board-size 19
```

The dataset files are intentionally not included until a 19x19 dataset is
available.
