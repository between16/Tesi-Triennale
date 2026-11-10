# Architecture

The project tests a Tiny Recursive Model (TRM) as a policy for deterministic
Go. It reuses the recursive reasoning core and adds a Go encoder, a policy
head, a rules environment, inference agents, and a Pygame interface.

## Model and inference

`TRM/models/go_model.py` is the shared `GoTRMModel` used by training, agents,
and evaluation. It combines:

- `TRM/models/go_encoder.py`, which encodes three board-history frames and the
  current player as seven channels of shape `[7, N, N]`;
- `TRM/models/go_trm.py`, the recursive reasoning core;
- `TRM/models/go_policy_head.py`, which emits `N*N + 1` logits: one per
  intersection and one for PASS (action index `N*N`).

The board size is checkpoint configuration, so the same model supports 9×9
and 19×19. Inference masks illegal moves before applying softmax. When choosing
a move, it samples uniformly among legal actions whose probability is at
least `0.98 * p_max`, where `p_max` is the best legal-action probability.
This is a relative 2% tolerance, not an absolute difference of two percentage
points. See [Experiments](./EXPERIMENTS.md) for the exact rule.

## Environment and interface

`TRM/game_env.py` owns board state, move legality, turn transitions, PASS
termination, and Tromp-Taylor/area scoring. `TRM/agents.py` adapts this
environment to human and checkpoint-backed TRM agents. The optional early
PASS-logit adjustment is an inference-time setting and does not change model
weights.

`TRM/app.py` provides the Pygame GUI. It selects a board-size-specific
checkpoint by default and also accepts an explicit checkpoint path.

## Data flow

The shared preprocessing script converts SGF games into examples and splits
by game to prevent positions from the same game leaking across splits. Input
features have shape `[examples, 7, N, N]`; targets are board actions or PASS.
Training, inference, and evaluation share the same model class and checkpoint
configuration.

See [Datasets](./DATASETS.md), [Training](./TRAINING.md), and
[Evaluation](./EVALUATION.md) for operational details.
