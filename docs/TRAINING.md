# Training

Install the project dependencies from the repository root with `pip install -r
requirements.txt` or `uv sync`. Prepare local datasets as described in
[Datasets](./DATASETS.md), then run the following commands from `TRM/`.

## Standard runs

```bash
python3 train.py --board-size 9 \
  --dataset-dir 9x9_go/go_dataset_go9 --epochs 20
```

For 19×19, first prepare the recommended deterministic 10,000-game sample:

```bash
python3 train.py --board-size 19 \
  --dataset-dir 19x19_go/go_dataset_professional_10k --epochs 5
```

These are supervised next-move imitation runs; Top-1 measures agreement with
the recorded move, not complete-game strength. Training selects the data
folder for each board size when `--dataset-dir` is omitted. NPY split
directories are preferred automatically when both NPY and NPZ are present.

Use `--max-steps` for a smoke run or `--max-hours` to bound a training run.
`--resume` restores the latest checkpoint. The best validation checkpoint is
saved separately from the resumable checkpoint:

- 9×9: `checkpoints/best_model.pt`
- 19×19: `checkpoints/19x19_go/best_model.pt`

The 19×19 `best_model.pt` is the inference checkpoint; `latest_model.pt`
contains resumable training state. Generated datasets and resumable
checkpoints are local artifacts and are excluded from Git.

## Validation and test metrics

The trainer reports train and validation Top-1/Top-3 metrics. The evaluator
reports held-out test-set metrics; its default split follows the checkpoint
board size and prefers NPY data. Validation Top-1 and test Top-1 can differ
because they are measured on separate splits.

See [Evaluation](./EVALUATION.md) for test evaluation commands and
game-strength protocols.
