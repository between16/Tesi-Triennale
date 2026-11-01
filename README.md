# Goal
The goal is to test if Tiny Recursive Model (TRM) is suitable for solving deterministic problems like playing Go, changing only what is necessary and keeping intact the core recursive architecture.

The Go training pipeline supports square boards of configurable size. The repository includes the trained 9x9 model. A separate 19x19 pipeline is also available; see the practical training instructions below.

# Setup
In order to replicate the project on your machine you need to install all the necessary libraries and you can do that in two ways:

1. If you use pip:
```bash
pip install -r requirements.txt
```
2. If you use uv:
```bash
uv sync
```

# How to use:
If you want to **play the game**, the repository already contains a trained 9x9 model. After installing the libraries, run the app from the `TRM/` directory. The default is 9x9 and uses `checkpoints/best_model.pt`:
```bash
cd TRM/
python3 app.py --board-size 9
```

To play 19x19, select that board size; the app will load the matching checkpoint from `checkpoints/19x19_go/best_model.pt`:
```bash
python3 app.py --board-size 19
```

The mode selector in the GUI continues to cycle through all three existing modes: Human (Black) vs TRM (White), TRM (Black) vs Human (White), and TRM vs TRM. The selected board size applies to the whole game and is validated against the loaded checkpoint. You can also load a checkpoint directly with `--checkpoint`; if `--board-size` is also provided, the two sizes must match.

When the TRM selects a move, it randomizes uniformly only among legal actions whose predicted probability is within 2 percentage points of the best legal action. If no other action is that close, it selects the best action. This adds variety without sampling broadly from lower-probability moves; it also applies to greedy self-play in `go_pass_eval.py`.

The 19x19 launch command requires a trained 19x19 checkpoint. The repository currently contains only the 9x9 model. After training a 19x19 model, commit its small inference checkpoint at `TRM/checkpoints/19x19_go/best_model.pt`; the large training datasets are not needed for inference.

If you want to **prepare a dataset**, place SGF files under the matching raw-games directory and run preprocessing with the board size. By default, preprocessing reads 9x9 games from `9x9_go/go_raw_dataset/9x9_games/` and writes the train, validation, and test datasets to `9x9_go/go_dataset/`.
```bash
cd TRM/
python3 go_preprocessing.py --board-size 9
```

To prepare a full 19x19 dataset from already filtered SGF files under `19x19_go/go_raw_dataset/19x19_games/`, run:
```bash
python3 go_preprocessing.py --board-size 19
```

For the default source/output convention, preprocessing for either board size
uses `go_raw_dataset/<N>x<N>_games/` as input and `go_dataset/` as output;
changing `--board-size` selects the corresponding `N` folders and validates
the SGF board size. The same script can write either compressed NPZ files or
memory-mapped NPY split directories.

For local training, the recommended path uses the professional-player records in `19x19_go/go_raw_dataset/games/`, samples up to 10,000 games, and writes memory-mapped NPY splits into the standard 19x19 dataset folder. Run from `TRM/`:
```bash
python3 9x9_go/go_raw_dataset/extract_9x9.py --board-size 19 \
  --source 19x19_go/go_raw_dataset/games \
  --destination 19x19_go/go_raw_dataset/19x19_professional_games
python3 go_preprocessing.py --board-size 19 \
  --input-dir 19x19_go/go_raw_dataset/19x19_professional_games \
  --max-games 10000 --storage-format npy
```

The source contains 96,858 readable 19x19 games; after exact duplicate removal, 95,715 remain. Sampling is deterministic (seed 42), and the games are split 80/10/10 before preprocessing, so moves from one game never cross splits. The current 10,000-game sample produced 1,668,062 train examples, 208,594 validation examples, and 208,653 test examples. Two unreadable SGFs were skipped. These local datasets are excluded from Git because of their size.

The preprocessing script reconstructs missing terminal PASS moves only for fully processed games with an explicitly scored result. It does not add PASS moves to games ending by resignation, time loss, or forfeiture.

To **filter SGF files by board size** from an archive directory, use the extraction helper. The source path is relative to the current working directory:
```bash
cd TRM/
python3 9x9_go/go_raw_dataset/extract_9x9.py --board-size 9 --source ./games
python3 9x9_go/go_raw_dataset/extract_9x9.py --board-size 19 --source ./games
```

If you want to **train the model from scratch**, pass the same board size used to prepare the dataset. The 9x9 checkpoint is saved to `checkpoints/best_model.pt`; models for other sizes are saved in a size-specific subdirectory such as `checkpoints/19x19_go/best_model.pt`.
```bash
cd TRM/
python3 train.py --board-size 9
```

For 9x9 and 19x19, the same training script and data loader are used. The
board-size option selects the model dimensions and the matching dataset and
checkpoint directories. The current 9x9 dataset is NPZ; the recommended
19x19 sample is NPY memory-mapped. The loader detects either format
automatically, so once each dataset is in its standard board-size folder, the
only required training option is `--board-size`:
```bash
python3 train.py --board-size 9 --epochs 20
python3 train.py --board-size 19 --epochs 5
```

The 19x19 defaults are batch size 16 and two data-loader workers; the 9x9 batch size remains 128. Use `--dataset-dir` to select a non-standard dataset location, `--dataset-format npy` or `--dataset-format npz` to override auto-detection, and `--max-hours` or `--max-steps` for a bounded run. `--epochs 5` means five complete passes over the train split. On the local RTX 4060, a 1,000-step benchmark measured about 480 examples/s with batch size 16. At that rate, one pass over the current 19x19 train split takes around 58 minutes; five passes are approximately 4 hours 50 minutes, before validation and checkpoint overhead. These are estimates, not guarantees. Training reports throughput, saves a resumable `latest_model.pt` every 500 steps, and saves the best validation model as `checkpoints/19x19_go/best_model.pt`.

The inference checkpoint contains only model weights and configuration and is about 3.3 MB of weights, well below GitHub's 100 MB per-file limit. `latest_model.pt` contains optimizer state and remains ignored; the best checkpoint is intentionally allowed by `.gitignore`. After training, add the inference checkpoint to the repository with:
```bash
git add TRM/checkpoints/19x19_go/best_model.pt
git commit -m "Add trained 19x19 Go model"
```
Once that checkpoint is committed, the repository can run 19x19 inference without the datasets. Until then, a fresh clone has no trained 19x19 model.

The policy head produces `N*N + 1` logits: one for each board intersection and one for PASS. The PASS action index is `N*N` (81 on 9x9 and 361 on 19x19). The board size is stored in each new checkpoint and is used automatically by inference and the GUI.

Training saves `best_model.pt` whenever validation loss improves during that training run. The training output's `Train Top-1` is measured on the training split, while `Val Top-1` and the checkpoint's `val_top1` are measured on the validation split. `go_eval.py` reports accuracy on the separate test split, so its Top-1 can legitimately differ from the checkpoint's validation Top-1. The training and validation metrics are aggregated per example, including a smaller final batch.

To **evaluate a trained model** on its corresponding test dataset:
```bash
python3 go_eval.py --checkpoint checkpoints/best_model.pt
```

Evaluation reads the board size from the checkpoint and automatically chooses
the matching test split. It accepts both `.npz` files and NPY split
directories. To evaluate the 19x19 model:
```bash
python3 go_eval.py --checkpoint checkpoints/19x19_go/best_model.pt
```

Evaluation metrics, predictions, and plots are saved under `evaluation_results/` (with a board-size-specific subdirectory for non-default board sizes).

The optional micro-training check uses the selected board size and loads either an
NPZ or NPY train split when available; if no dataset exists it uses synthetic
data:
```bash
python3 micro_train.py --board-size 9
python3 micro_train.py --board-size 19
```

# Dataset
The dataset source is: https://homepages.cwi.nl/~aeb/go/games/

The repository includes the processed 9x9 dataset. The 19x19 professional SGFs, generated NPY splits, and full NPZ datasets are local data artifacts and are excluded from Git. The `games/` directory contains the professional records; `jgdb/` is an additional, separate source that can be included when building the full 19x19 archive. The 9x9 and 19x19 splits are stored in their respective `TRM/<N>x<N>_go/go_dataset/` directories.

# Changes Made:
Below you can find the list of all the changes made to the original TRM:
## Added Repository
```bash
TRM/9x9_go
```
This folder contains the raw SGF games used for the 9x9 dataset and the processed train, validation, and test splits.

```bash
TRM/19x19_go
```
This folder contains the 19x19 raw-source layout and the matching dataset directory. Local professional SGFs and generated datasets are ignored by Git; the per-board folder layout lets shared scripts locate data using `--board-size`.

## Added Files
### TRM

```bash
TRM/go_preprocessing.py
```
This is the shared SGF-to-dataset pipeline for any square board size. `--board-size` selects the expected SGF size and standard input/output folders. `--input-dir` and `--output-dir` support alternate corpora and locations; `--max-games` creates a reproducible sample; `--storage-format npz` writes compressed archives and `--storage-format npy` writes memory-mappable split directories. Each input `X` has shape `[SIZE, 7, N, N]`; each target `y` has shape `[SIZE]` and contains a board action or PASS (`N*N`).

The data is split by game, so moves from a single SGF are not distributed across different splits. For fully processed games with an explicitly scored result, missing consecutive terminal passes are reconstructed to provide terminal examples.

```bash
TRM/9x9_go/go_raw_dataset/extract_9x9.py
```
Despite its historical filename, this extraction helper is board-size
agnostic. It recursively filters one or more SGF sources by the board size
recorded in each game, handles `.sgf` extensions case-insensitively, removes
byte-identical duplicates, and reports unreadable files.

```bash
TRM/go_dataset.py
```
This module provides the shared PyTorch dataset and dataloader. It loads
compressed NPZ data or memory-mapped `X.npy`/`y.npy` split directories,
validates board/action dimensions, and applies random rotations/reflections
to training examples.

```bash
TRM/micro_train.py
```
This script runs a 100-step single-batch overfit/smoke check. The board size
selects the matching model and dataset; it reads either supported dataset
format or uses synthetic data if no dataset is present.

```bash
TRM/train.py
```
This shared trainer configures and runs supervised next-move training for
both board sizes. `--board-size` selects the model dimensions and standard
dataset/checkpoint paths; optional arguments control data format, batch size,
epoch or wall-clock/step limits, validation, and resumable checkpoints.

```bash
TRM/game_env.py
```
This file implements the Go environment and handles:
- Storing the board state and the history of the last 3 turns (required by the `GoEncoder`).
- Checking legal moves and generating the Legal Move Mask.
- Managing turn transitions and PASS moves.
- Calculating the final score using Tromp-Taylor/area scoring.

The board size is configurable. PASS is always legal during an active game, and two consecutive passes end the game.

```bash
TRM/agents.py
```
This file implements `HumanAgent` and `TRMAgent`. The TRM agent loads the board size from its checkpoint, converts the environment state into model features, masks illegal moves, and randomizes only among legal moves within 2 percentage points of the top probability. It rejects environments whose board size does not match the checkpoint.

```bash
TRM/go_eval.py
```
This file evaluates supervised next-move prediction on the test split
associated with the selected checkpoint. Board size is read from the
checkpoint; the evaluator accepts NPZ files and memory-mapped NPY split
directories. It saves metrics, predictions, and diagnostic plots under the
size-specific `evaluation_results/` folder. Test Top-1 is not expected to
match validation Top-1 exactly.

```bash
TRM/test_go_preprocessing.py
```
This regression suite checks SGF extraction, game filtering, position counts,
NPZ and memory-mapped NPY creation, and automatic split-format selection.

```bash
TRM/go_pass_eval.py
```
This file runs self-play using a selected model checkpoint to inspect
termination behavior, PASS frequency, and game statistics. The checkpoint
determines the board size. Results are saved under the matching
`pass_eval_results/` folder.

```bash
TRM/app.py
```
This file provides the Pygame GUI. `--board-size 9|19` selects the standard
size-specific inference checkpoint; `--checkpoint` can select a different
file. It supports Human vs TRM (either color) and TRM vs TRM, configurable
move delay, and pausing AI-vs-AI games. The selected checkpoint determines
the model's board dimensions.
![GUI](./TRM/assets/GUI.png)

### TRM/models
```bash
TRM/models/go_encoder.py
```
This file encodes a board tensor `[Batch, Channels, N, N]` into one token per intersection `[Batch, N*N, Hidden_size]`. A padded convolution preserves the board dimensions while extracting local features, and learnable positional embeddings provide each token with its board location.

```bash
TRM/models/go_policy_head.py
```
This file maps the TRM output tokens to policy logits. It produces `N*N` logits for board moves and one additional logit for PASS, for a total of `N*N + 1`.

```bash
TRM/models/common.py
TRM/models/layers.py
```
These shared model primitives provide initialization, normalization,
attention, rotary embeddings, and feed-forward layers used by the Go TRM.

## Modified files
```bash
TRM/models/go_trm.py
```
This module combines the Go encoder with the recursive reasoning core. The
configured board size determines the sequence length (`N*N`), input shape,
and policy vocabulary. Checkpoints created before board-size configuration
are interpreted as 9x9.

The shared GUI and inference agent are implemented in `TRM/agents.py`; the
agent builds board-history features, masks illegal moves, and loads the
board dimensions from the checkpoint. `TRM/game_env.py` supplies the rules and
legal-move checks consumed by both the GUI and self-play evaluation.

# Testing
`test_game_env.py` checks board rules and legal-move behavior.
`test_go_board_size.py` checks model output dimensions, GUI checkpoint
selection, board symmetries, and action-selection helpers for both board
sizes. `test_go_preprocessing.py` checks SGF filtering/deduplication,
position counting, NPZ/NPY dataset writing, and dataset loading.

Run the unit tests from the repository root with:
```bash
python3 -m pytest TRM/test_game_env.py TRM/test_go_board_size.py TRM/test_go_preprocessing.py
```

# Pipeline
PIPELINE FOR TRAINING PHASE:
![Pipeline per training](/TRM/assets/pipeline_for_trainig.png)

PIPELINE FOR INFERENCE PHASE:
![Pipeline for inference](/TRM/assets/pipeline_inference.png)
