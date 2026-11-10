# Evaluation

The repository has two distinct evaluation methods: supervised next-move
accuracy on held-out SGFs and complete-game matches against fixed opponents.
They measure different properties and should not be conflated.

## Held-out next-move prediction

From `TRM/`, evaluate the matching test split:

```bash
python3 model_evaluation/go_eval.py --checkpoint checkpoints/best_model.pt \
  --test-data 9x9_go/go_dataset_go9/test
python3 model_evaluation/go_eval.py \
  --checkpoint checkpoints/19x19_go/best_model.pt \
  --test-data 19x19_go/go_dataset_professional_10k/test
```

The test path may be an NPY split directory or an NPZ archive. If omitted,
the evaluator selects the documented split for the checkpoint's board size.
Metrics, predictions, and plots are written under
`model_evaluation/results/`.

The previously verified 9×9 `go_dataset_go9` test split contains 41,380
examples; its Top-1/Top-3/Top-5 scores were 44.74%, 70.93%, and 82.31%.
These are imitation-accuracy results, not estimates of win rate.

## Complete-game strength

`model_evaluation/go_match.py` runs color-balanced matches against a local
KataGo GTP engine. The configured rules match `GoGameEnv`: area scoring, 7.5
komi, positional superko, suicide forbidden, no button or tax, and termination
after two consecutive passes. Rules are checked before a match starts.

KataGo binaries, runtime libraries, and network files are external local
resources; their source URLs, licenses, and checksums are recorded in
`TRM/model_evaluation/config/katago-local-resources.json`. Match results and
the runner's configuration are saved as CSV. Use a fixed visit budget and
alternate colors for a repeatable comparison:

```bash
python3 model_evaluation/go_match.py --board-size 9 \
  --katago /path/to/katago \
  --katago-config /path/to/default_gtp.cfg \
  --katago-model /path/to/kata9x9.bin.gz \
  --max-visits 64 --games 40 \
  --output model_evaluation/results/9x9_go/matches-64-visits.csv
```

The random-agent runner is a weak sanity-check baseline, not a calibrated Go
rating. HumanSL profile results are also relative to the exact KataGo model,
profile, and sampling settings; they are not official KGS/AGA ranks or
absolute Elo anchors.

See [Experiments](./EXPERIMENTS.md) for the project’s recorded match results,
limitations, and PASS-policy study.
