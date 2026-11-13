# Go adaptation of Tiny Recursive Models

This directory contains a Go-playing adaptation of the Tiny Recursive Model
(TRM), with 9×9 and 19×19 datasets, training, inference, evaluation, and a
Pygame interface. Start with the repository [README](../README.md) and the
detailed [project documentation](../docs/).

## Credits and license

The recursive model design is based on **Less is More: Recursive Reasoning
with Tiny Networks** by Alexia Jolicoeur-Martineau ([paper](https://arxiv.org/abs/2510.04871),
[original code](https://github.com/SamsungSAILMontreal/TinyRecursiveModels)).
The earlier recursive-reasoning lineage includes the **Hierarchical Reasoning
Model** by Guan Wang et al. ([paper](https://arxiv.org/abs/2506.21734),
[code](https://github.com/sapientinc/HRM)).

The original model descriptions and code-derived components retain their
applicable notices in [`LICENSE`](./LICENSE). Please preserve those notices
when redistributing substantial portions of the original work.

## Project layout

- `agents.py`, `game_env.py`, and `app.py`: Go agents, rules, and GUI.
- `go_preprocessing.py`, `go_dataset.py`, and `train.py`: dataset pipeline
  and supervised training.
- `models/`: shared Go encoder, recursive model, and policy head.
- `model_evaluation/`: test-set metrics and complete-game evaluation.
- `tests/`: Go environment, preprocessing, training, and inference tests.

The original ARC, Sudoku, and Maze preparation/training code is not part of
this Go project. The upstream papers and source repository remain credited
above; the applicable license is preserved in this directory.

## Policy-guided Negamax search

The Pygame interface can optionally wrap the trained TRM policy in a bounded
Negamax search with alpha-beta pruning. The TRM ranks candidate moves; the
search explores a limited set of replies and evaluates leaf positions using
area score plus a heuristic penalty for groups with few liberties. No model
retraining or checkpoint changes are required.

From this directory, start the 19x19 GUI with:

```bash
uv run python app.py --board-size 19 --search negamax
```

The baseline policy-only agent remains the default. Search options are:

- `--search policy|negamax`: choose the baseline agent or Negamax wrapper.
- `--search-depth 2`: number of plies searched (default: 2).
- `--search-candidates 3`: maximum policy-ranked board moves per node.
  The default is 5 on 9x9 and 3 on 19x19. Tactical atari responses are added
  even when they fall outside this policy-ranked set.

This is a deliberately bounded, experimental search. Its heuristic is not a
learned value function, and restricting candidate moves means it does not
provide an optimal-play guarantee. Compare it against `--search policy` using
the same checkpoints and game settings.
