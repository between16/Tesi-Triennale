# Policy-guided Negamax

The GUI can wrap the trained TRM policy in a bounded Negamax search with
alpha-beta pruning. The policy supplies move probabilities and ranks the
candidate moves; it is not replaced by a value model, and no retraining or
checkpoint changes are needed.

## Run

Run these commands from `TRM/`, where `app.py` and the standard checkpoints
are located:

```bash
python app.py --board-size 19 --search negamax
```

For an explicit depth-two search with three policy-ranked board moves per
node:

```bash
python app.py --board-size 9 --search negamax \
  --search-depth 2 \
  --search-candidates 5
```

```bash
python app.py --board-size 19 --search negamax \
  --search-depth 2 \
  --search-candidates 3
```

With `uv`, run `uv run python app.py ...` from the same directory. The
board-size option selects its matching standard checkpoint unless
`--checkpoint` is supplied.

## Search options

- `--search policy|negamax` selects direct policy play or policy-guided search.
  Policy play is the default.
- `--search-depth` is measured in plies: depth 2 searches the current move and
  the opponent's reply. The default is 2.
- `--search-candidates` caps the number of policy-ranked board moves considered
  at each node. It defaults to 5 on 9×9 and 3 on 19×19. Additional tactical
  moves may be considered beyond this number.

At each node, the search orders legal moves using the policy probabilities.
It also includes legal liberties of groups in atari, allowing the search to
consider saving its own group or capturing an opponent's group even when the
policy ranks that move below the candidate limit. PASS is included when it is
among the top-ranked actions or when the opponent has just passed, so the
search can end the game with two consecutive passes. If no board moves are
available, PASS is used.

Each branch uses a copy of the game environment, leaving the live game state
unchanged. Terminal positions use the environment's area score and komi, with
a bonus so a terminal win or loss takes precedence over a non-terminal
heuristic. Non-terminal leaves use area-score difference plus a heuristic
penalty for groups with few liberties. The heuristic is from the perspective
of the player to move; it is not a learned Go value function.

## Validation and limitations

The focused tests cover legal moves, root-state immutability, cloning,
perspective handling, PASS behavior, and immediate atari saves and captures:

```bash
python3 -m pytest TRM/tests/test_negamax.py
```

This is an experimental, deliberately bounded search. Candidate pruning can
omit good moves, and the heuristic is not strong enough to guarantee optimal
play. A successful search or a legal move does not establish playing strength;
compare complete games against the policy-only agent under identical
checkpoints and settings for that purpose.

## Negamax versus KataGo

The complete-game runner supports the same search selector. Run it from
`TRM/`; keep all opponent and game parameters fixed and change only
`--search` to compare policy-only play with Negamax:

```bash
python3 model_evaluation/go_match.py \
  --board-size 19 \
  --katago /path/to/katago \
  --katago-config /path/to/default_gtp.cfg \
  --katago-model /path/to/kata19x19.bin.gz \
  --games 100 \
  --max-visits 40 \
  --seed 42 \
  --search negamax \
  --search-depth 2 \
  --search-candidates 3 \
  --output model_evaluation/results/19x19_go/negamax-40v-100g.csv
```

`--search policy` is the default and remains the baseline. Each run produces a
CSV with a row per game and a separate JSON report containing the invocation,
effective parameters, model hashes, color-balanced results, and search
statistics. The runner preserves prior files by choosing a timestamped path if
the requested output already exists. See [Evaluation](./EVALUATION.md) for
complete 9×9 and 19×19 commands and report contents.
