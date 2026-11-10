# Datasets

SGFs and generated datasets are not required for inference. The 9×9 source
corpus is included in the repository; the 19×19 source archive and all
generated split directories are local artifacts ignored by Git. Prepare the
datasets before training or test-set evaluation.

## Sources and snapshot identity

### 9×9

The project uses the **Go Quest – Top 50 players** SGF corpus at
`TRM/9x9_go/go_raw_dataset/go9/`, which is included in the repository. The
original [28 December 2015 announcement](https://www.eugeneweb.com/pipermail/computer-go/2015-December/008353.html)
describes the 9×9 Top 50 collection (up to 200 games per player) and links to
[`go9.tgz`](https://www.dropbox.com/s/abpzmqrw7gyvlzt/go9.tgz). The announcement
reports 8,607 games, while the repository snapshot contains 8,606. Therefore
the public archive is the closest traceable source, but it is not verified to
be byte-identical to the snapshot used for the reported results.

The checked-in snapshot's manifest SHA-256 is:
`33241a305c10323c90f7bec3f42655050b7f902008f350b5709a62654bbb1b5a`

### 19×19

The original professional archive is the
[CWI Database of Go Games](https://homepages.cwi.nl/~aeb/go/games/). Its
download page offers `games.tgz` and `games.7z`; extract the archive so the
professional SGFs are under `TRM/19x19_go/go_raw_dataset/games/`. The project
uses the `games/` archive only, not the additional `jgdb/` collection.

The source snapshot used here contains 97,565 SGF files. Filtering readable
19×19 games and removing byte-identical duplicates yields 95,715 files. CWI
does not identify this collection with a version tag; the snapshot manifest
SHA-256 is:
`b075c1500f4b8dc75b51dc573ce98e3695d327429868de5001bc29275ff448e8`

The manifest hash is SHA-256 over sorted relative paths and the SHA-256 digest
of each SGF. A changed archive should be recorded as a new snapshot rather
than described as the same dataset version.

## Prepare the datasets

Run commands from `TRM/`. The extraction helper checks SGF readability and
board size, removes byte-identical duplicates, and writes a filtered
intermediate corpus.

### 9×9

```bash
python3 9x9_go/go_raw_dataset/extract_9x9.py --board-size 9 \
  --source 9x9_go/go_raw_dataset/go9 \
  --destination 9x9_go/go_raw_dataset/go9_filtered
python3 go_preprocessing.py --board-size 9 \
  --input-dir 9x9_go/go_raw_dataset/go9_filtered \
  --output-dir 9x9_go/go_dataset_go9 \
  --storage-format npy
```

### 19×19

```bash
python3 9x9_go/go_raw_dataset/extract_9x9.py --board-size 19 \
  --source 19x19_go/go_raw_dataset/games \
  --destination 19x19_go/go_raw_dataset/19x19_professional_games
python3 go_preprocessing.py --board-size 19 \
  --input-dir 19x19_go/go_raw_dataset/19x19_professional_games \
  --output-dir 19x19_go/go_dataset_professional_10k \
  --max-games 10000 \
  --storage-format npy
```

Preprocessing uses seed 42 to sample at most 10,000 games and split by game
into 80/10/10 train, validation, and test sets. The generated NPY format
stores `X.npy` and `y.npy` in each split directory and supports memory mapping.
The alternative compressed format writes `train.npz`, `val.npz`, and
`test.npz`.

## Shapes and observed split sizes

Raw games have variable lengths. Processed features are `float32` arrays with
shape `[examples, 7, N, N]`; targets are `int64` action indices with shape
`[examples]`.

| Board | Corpus used | Train examples | Validation examples | Test examples | Total |
|---|---:|---:|---:|---:|---:|
| 9×9 | 8,606 games | 331,251 | 41,500 | 41,380 | 414,131 |
| 19×19 | up to 10,000 of 95,715 unique games | 1,668,062 | 208,594 | 208,653 | 2,085,309 |

Counts describe the currently verified snapshots and can change when source
files or processing inputs change. The 19×19 generated dataset is a sampled
subset of the professional archive, not the complete archive.
