# 19×19 Go datasets

The professional archive is the **CWI Database of Go Games**
([download page](https://homepages.cwi.nl/~aeb/go/games/), with `games.tgz`
and `games.7z` archives). Its SGF files use nested layouts and may contain
different board sizes. Extract the archive so the source is
`go_raw_dataset/games/`. The separate
`go_raw_dataset/jgdb/` source is not part of the practical training dataset
described below.

## Recommended local training dataset

For a practical local training run, use only the professional-player
`games/` source. The snapshot used by this project contains 97,565 SGF files;
filtering for valid 19×19 games and removing exact duplicates produces 95,715
unique games. The commands below create deterministic, game-level
train/validation/test splits from at most 10,000 games. Run from `TRM/`:

```bash
python 9x9_go/go_raw_dataset/extract_9x9.py --board-size 19 \
  --source 19x19_go/go_raw_dataset/games \
  --destination 19x19_go/go_raw_dataset/19x19_professional_games
python go_preprocessing.py --board-size 19 \
  --input-dir 19x19_go/go_raw_dataset/19x19_professional_games \
  --output-dir 19x19_go/go_dataset_professional_10k \
  --max-games 10000 \
  --storage-format npy
```

The sample is shuffled with seed 42 before the 80/10/10 split, so each game
belongs to exactly one split. NPY split directories contain `X.npy` and
`y.npy` arrays and are read via memory mapping. They do not require the entire
dataset to be loaded into RAM. Raw SGFs have variable numbers of moves and
therefore no single tensor shape. The current/previously generated sample
contains 1,668,062 train, 208,594 validation, and 208,653 test examples
(2,085,309 total); `X` has shape `[examples, 7, 19, 19]` and dtype
`float32`, while `y` has shape `[examples]` and dtype `int64`. Counts may
change if the input files or selected sample change. The archive has no
version tag; its snapshot fingerprint is recorded in
[`../../docs/DATASETS.md`](../../docs/DATASETS.md).

Start with a short smoke test, then launch the run with the available
eight-hour budget:

```bash
python train.py --board-size 19 \
  --dataset-dir 19x19_go/go_dataset_professional_10k --max-steps 100
python train.py --board-size 19 \
  --dataset-dir 19x19_go/go_dataset_professional_10k --max-hours 8 --seed 42
```

The NPY splits are written to
`19x19_go/go_dataset_professional_10k/{train,val,test}/`. Pass
`--dataset-dir` when using this dedicated output directory. If no custom
directory is supplied, training detects NPY directories (and otherwise falls
back to NPZ files) under `go_dataset_professional_10k/`.
If that directory is absent, training falls back to the existing legacy
`go_dataset/` directory, which is also the dataset path used by the earlier
19×19 run in this repository.
For 19×19, the default batch size is 16. Validation uses the full validation
split unless `--validation-batches` is specified. A resumable checkpoint is
saved every 500 steps and at the end of each epoch. To resume this board size,
use `--board-size 19 --resume`; this loads
`checkpoints/19x19_go/latest_model.pt`, not the 9×9 checkpoint. The checkpoint
stores its seed and RNG state, model/optimizer/scheduler state, complete
configuration, and dataset metadata. Mid-epoch resumes continue with the
first batch not yet saved; completed epochs continue with the next epoch.

`--epochs` is the total target epoch count. For example, to continue a
20-epoch run through epoch 50:

```bash
python train.py --board-size 19 --epochs 50 --resume
```

The saved dataset directory and training settings are restored when omitted.
Resume rejects a different dataset or board size rather than silently mixing
runs. Older checkpoints lack dataset metadata; provide the original
`--dataset-dir` if it differs from the current default. `--seed` defaults to
42 for fresh runs. Checkpoints created before RNG state was recorded can
still resume their model and optimizer, but cannot continue the exact random
sequence. Extending the epoch target keeps the saved scheduler horizon and
continues at its minimum learning rate once reached. Training prints the
observed throughput; use it to estimate how much data and how many epochs fit
on the local GPU.
The time limit is checked between batches, and final validation/checkpointing
can extend it slightly.
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

To combine the professional and JGDB records, filter them into a separate
folder and preprocess with the compressed NPZ format:

```bash
python 9x9_go/go_raw_dataset/extract_9x9.py --board-size 19 \
  --source 19x19_go/go_raw_dataset/games 19x19_go/go_raw_dataset/jgdb \
  --destination 19x19_go/go_raw_dataset/19x19_all_games
python go_preprocessing.py --board-size 19 \
  --input-dir 19x19_go/go_raw_dataset/19x19_all_games \
  --output-dir 19x19_go/go_dataset_full --storage-format npz
```

This full dataset is very large. NPZ loading decompresses the arrays into
memory; use the sampled NPY splits above for local training. The same
preprocessing and memory-mapped format work for 9×9 by changing
`--board-size 19` to `--board-size 9` and selecting the corresponding source
directories. Raw archives and generated datasets are ignored by Git.
