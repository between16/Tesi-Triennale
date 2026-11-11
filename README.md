# Tiny Recursive Model for Go

This project explores whether a Tiny Recursive Model (TRM) can learn to play
deterministic games, using Go as a case study while preserving the recursive
reasoning core.

## Architecture

The Go system combines a board encoder, the TRM reasoning core, a policy head,
a rules environment, inference agents, and a Pygame interface. Checkpoints
support both 9×9 and 19×19 boards. See [Architecture](./docs/ARCHITECTURE.md).

## Features

- Play Human vs TRM (either color) or TRM vs TRM.
- Select 9×9 or 19×19 boards and matching checkpoints.
- Preprocess SGF corpora, train models, and evaluate move prediction or
  complete-game strength.
- Run seeded, reproducible training and resume 9×9/19×19 checkpoints; see
  [Training](./docs/TRAINING.md).

## Requirements and installation

Python 3.11 or newer is required. Install dependencies from the repository
root with either:

```bash
pip install -r requirements.txt
```

or:

```bash
uv sync
```

## Launch the GUI

Run from `TRM/`. The default 9×9 checkpoint is selected automatically:

```bash
cd TRM
python3 app.py --board-size 9
```

For 19×19:

```bash
python3 app.py --board-size 19
```

## Tests

Run the unit tests from the repository root:

```bash
python3 -m pytest
```

## Detailed documentation

- [Architecture](./docs/ARCHITECTURE.md): model, game environment, inference,
  and GUI.
- [Datasets](./docs/DATASETS.md): source archives, snapshot identity,
  preparation, and dataset formats.
- [Training](./docs/TRAINING.md): training commands, checkpoints, and metrics.
- [Evaluation](./docs/EVALUATION.md): test-set evaluation and complete-game
  match protocols.
- [Experiments](./docs/EXPERIMENTS.md): move selection, PASS adjustment, and
  recorded results.
