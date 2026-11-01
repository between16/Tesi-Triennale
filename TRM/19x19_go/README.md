# 19×19 Go datasets

The raw archives include professional records in `go_raw_dataset/games/` and
additional games in `go_raw_dataset/jgdb/`. The SGF files use nested layouts
and may contain different board sizes.

## Recommended local training dataset

For a practical local training run, start with the professional-player
`games/` source. The directory currently contains 96,858 readable 19×19
records; exact duplicates reduce that to 95,715 unique games. The commands
below filter it and create deterministic, game-level train/validation/test
splits from at most 10,000 games. Run from `TRM/`:

```bash
python 9x9_go/go_raw_dataset/extract_9x9.py --board-size 19 \
  --source 19x19_go/go_raw_dataset/games \
  --destination 19x19_go/go_raw_dataset/19x19_professional_games
python go_preprocessing.py --board-size 19 \
  --input-dir 19x19_go/go_raw_dataset/19x19_professional_games \
  --max-games 10000 --storage-format npy
```

The sample is shuffled with seed 42 before the 80/10/10 split, so each game
belongs to exactly one split. NPY split directories contain `X.npy` and
`y.npy` arrays and are read via memory mapping. They do not require the entire
dataset to be loaded into RAM. The generated sample contains 1,668,062 train,
208,594 validation, and 208,653 test examples (two unreadable SGFs were
skipped).

Start with a short smoke test, then launch the run with the available
eight-hour budget:

```bash
python train.py --board-size 19 --max-steps 100
python train.py --board-size 19 --max-hours 8
```

The NPY splits are written to `go_dataset/{train,val,test}/`, the standard
board-size-specific dataset directory. Once a dataset is prepared, no
`--dataset-dir` or `--dataset-format` argument is needed; training detects
NPY directories (and otherwise falls back to NPZ files) in that location.
For 19×19, the default batch size is 16. Validation is limited to 200 batches
by default to leave time for training. A resume checkpoint is saved every
500 steps and when the time limit is reached; `--resume` loads it. If it was
saved mid-epoch, resuming restarts that epoch. The best validation model is
also saved at each validation, including a final bounded validation at the
time limit. Training prints the observed throughput; use it to estimate how
much data and how many epochs fit on the local GPU. The time limit is checked
between batches, and final validation/checkpointing can extend it slightly.
An initial 1,000-step benchmark on the current RTX 4060 measured about 480
examples/s, suggesting roughly eight passes over the train split in eight
hours before overhead; treat this as a preliminary estimate rather than a
guarantee.

The inference-ready checkpoint is
`TRM/checkpoints/19x19_go/best_model.pt`. It stores the model weights and
configuration (about 3.3 MB of weights), so it is well below GitHub's 100 MB
file limit. This file is intentionally not ignored; add and commit it after
training to make 19×19 inference available immediately after cloning. The
larger `latest_model.pt` resume checkpoint remains ignored.

## Full archive

To combine the professional and JGDB records, first extract them into the
general filtered folder and preprocess with the default compressed NPZ
format:

```bash
python 9x9_go/go_raw_dataset/extract_9x9.py --board-size 19 \
  --source 19x19_go/go_raw_dataset/games 19x19_go/go_raw_dataset/jgdb
python go_preprocessing.py --board-size 19
```

This full dataset is very large. NPZ loading decompresses the arrays into
memory; use the sampled NPY splits above for local training. The same
preprocessing and memory-mapped format work for 9×9 by changing
`--board-size 19` to `--board-size 9` and selecting the corresponding source
directories. Raw archives and generated datasets are ignored by Git.
