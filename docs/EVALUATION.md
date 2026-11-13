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
after two consecutive passes. Rules are checked before a match starts. The
target can use either direct policy selection or the policy-guided Negamax
search; per-game search statistics are recorded in the results.

KataGo binaries, runtime libraries, and network files are external local
resources; their source URLs, licenses, and checksums are recorded in
`TRM/model_evaluation/config/katago-local-resources.json`. Every run writes
the per-game CSV and a new JSON report alongside it. The report records the
invocation and effective parameters, SHA-256 hashes of the checkpoint,
KataGo executable, networks, config and rules, KataGo version, source revision,
outcomes by target color, incomplete games, termination and pass statistics,
Negamax search counts, and a 95% Wilson interval for win rate. Existing output
files are not overwritten: if a requested path already exists, a UTC run
suffix is added to the CSV and default report paths; an existing explicit
`--report-output` path is rejected. The report's `comparison_key` groups runs
with identical board, opponent, models, rules, seed and budgets while
excluding search mode and its depth/candidate settings, to assist
policy-versus-Negamax comparisons.
The seed initializes both the TRM action sampler and KataGo's search/neural
random seeds. Pass `--report-output` to select a JSON path; otherwise the
report is named after the CSV with a `.report.json` suffix. Paths stored in
reports and CSVs are repository-relative (external files are identified by
filename and SHA-256), so the artifacts can be shared across machines. Retain
both files. The JSON report identifies the root-relative path base explicitly;
the invocation's working directory is recorded separately.

Use the same settings, seed and balanced colors to compare policy-only play
against Negamax. Select the opponent network that matches the board size:

```bash
python3 model_evaluation/go_match.py \
  --board-size 9 \
  --katago model_evaluation/katago-local/engine/squashfs-root/usr/bin/katago \
  --katago-config model_evaluation/katago-local/engine/default_gtp.cfg \
  --katago-model model_evaluation/katago-local/downloads/kata9x9-b18c384nbt-20231025.bin.gz \
  --katago-human-model model_evaluation/katago-local/downloads/b18c384nbt-humanv0.bin.gz \
  --human-profile rank_20k \
  --games 100 \
  --max-visits 40 \
  --seed 42 \
  --pass-logit-penalty 3.5 \
  --pass-penalty-before-stones 16 \
  --search policy \
  --output model_evaluation/results/9x9_go/policy-rank20k-40v-100g.csv

python3 model_evaluation/go_match.py \
  --board-size 19 \
  --katago model_evaluation/katago-local/engine/squashfs-root/usr/bin/katago \
  --katago-config model_evaluation/katago-local/engine/default_gtp.cfg \
  --katago-model model_evaluation/katago-local/downloads/kata1-tf3-b11c768-s12382M-d6533M.bin.gz \
  --katago-human-model model_evaluation/katago-local/downloads/b18c384nbt-humanv0.bin.gz \
  --human-profile rank_20k \
  --games 100 \
  --max-visits 40 \
  --seed 42 \
  --pass-logit-penalty 3.5 \
  --pass-penalty-before-stones 16 \
  --search policy \
  --output model_evaluation/results/19x19_go/policy-rank20k-40v-100g.csv
```

For Negamax, replace `--search policy` with `--search negamax --search-depth 2`.
Candidate limits default to 5 on 9×9 and 3 on 19×19; set
`--search-candidates` explicitly when matching a chosen experiment. For a fair
comparison keep all other values identical, including checkpoint, KataGo
networks, HumanSL profile, visit budget, seed, rules and pass penalty. The
comparison key helps pair runs, but is not itself a statistical test.

The random-agent runner is a weak sanity-check baseline, not a calibrated Go
rating. HumanSL profile results are relative to the exact KataGo model,
profile, and sampling settings; they are not official KGS/AGA ranks or
absolute Elo anchors.

See [Experiments](./EXPERIMENTS.md) for the project’s recorded match results,
limitations, and PASS-policy study.
