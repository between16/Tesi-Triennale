# Training

Install the project dependencies from the repository root with `pip install -r
requirements.txt` or `uv sync`. Prepare local datasets as described in
[Datasets](./DATASETS.md), then run the following commands from `TRM/`.

## Standard runs

```bash
python3 train.py --board-size 9 \
  --dataset-dir 9x9_go/go_dataset_go9 --epochs 20 --seed 42
```

For 19×19, first prepare the recommended deterministic 10,000-game sample:

```bash
python3 train.py --board-size 19 \
  --dataset-dir 19x19_go/go_dataset_professional_10k --epochs 5 --seed 42
```

These are supervised next-move imitation runs; Top-1 measures agreement with
the recorded move, not complete-game strength. Training selects the data
folder for each board size when `--dataset-dir` is omitted. NPY split
directories are preferred automatically when both NPY and NPZ are present.
For 19×19, if `go_dataset_professional_10k/` is absent, an existing legacy
`go_dataset/` directory is used instead.

Use `--max-steps` for a smoke run or `--max-hours` to bound a training run.
`--seed` defaults to 42 and seeds Python, NumPy, PyTorch, CUDA, and the
training data loader. The data order and random board symmetries are also
reproducible across worker counts and checkpoint resumes, provided the same
software, hardware, and dataset are used.

`--resume` restores the latest checkpoint for the selected board size. The
`--epochs` value is the **total target**, not the number of additional epochs.
For example, after a completed 20-epoch 9×9 run, this resumes at epoch 21 and
continues through epoch 50:

```bash
python3 train.py --board-size 9 --epochs 50 --resume
```

For 19×19, use `--board-size 19 --epochs 50 --resume`; the matching
`checkpoints/19x19_go/latest_model.pt` is used. If the run used a custom
dataset directory, it is restored from the checkpoint when `--dataset-dir` is
omitted. A changed dataset, board size, or optimizer/scheduler settings is
rejected rather than silently mixed with the saved run. The checkpoint stores
the model and optimizer/scheduler states, seed and RNG states, full training
configuration, epoch/batch position, and dataset paths, split sizes, shapes,
and file metadata. An interruption within an epoch resumes from the next
unsaved batch; a completed epoch resumes from the following epoch. Extending
the requested epoch count does not restart or lengthen the saved scheduler
schedule: after that schedule reaches its minimum learning rate, training
continues at that minimum.

The best validation checkpoint is saved separately from the resumable
checkpoint:

- 9×9: `checkpoints/best_model.pt`
- 19×19: `checkpoints/19x19_go/best_model.pt`
- 9×9 resume state: `checkpoints/latest_model.pt`
- 19×19 resume state: `checkpoints/19x19_go/latest_model.pt`

The 19×19 `best_model.pt` is the inference checkpoint; `latest_model.pt`
contains resumable training state. Generated datasets and resumable
checkpoints are local artifacts and are excluded from Git. Checkpoints
created before RNG-state tracking was added remain resumable, but cannot
reproduce the exact pre-interruption random sequence. Older checkpoints also
lack dataset metadata, so pass the same `--dataset-dir` used for that run if
it is not the resolved board-size default.

## Validation and test metrics

The trainer reports train and validation Top-1/Top-3 metrics. The evaluator
reports held-out test-set metrics; its default split follows the checkpoint
board size and prefers NPY data. Validation Top-1 and test Top-1 can differ
because they are measured on separate splits.

See [Evaluation](./EVALUATION.md) for test evaluation commands and
game-strength protocols.
