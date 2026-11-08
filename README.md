# Goal
The goal is to test if Tiny Recursive Model (TRM) is suitable for solving deterministic problems like playing Go, changing only what is necessary and keeping intact the core recursive architecture.

The Go training pipeline supports square boards of configurable size. The repository includes trained 9x9 and 19x19 checkpoints; see the practical training instructions below.

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

The 19×19 launch command requires a trained 19×19 checkpoint at `TRM/checkpoints/19x19_go/best_model.pt`; the large training datasets are not needed for inference.

## End-to-end Go training pipeline

Run the commands from the `TRM/` directory after installing the project
dependencies. The extraction helper filters SGF records by board size and
removes exact duplicate files. Preprocessing creates game-level 80/10/10
train/validation/test splits in NPY format; each split contains `X.npy` and
`y.npy` and can be loaded by memory mapping. The dedicated output folders
below keep these datasets separate from any existing splits.

### 9×9

The `go9/` directory contains the new 9×9 SGF corpus:

```bash
cd TRM/
python3 9x9_go/go_raw_dataset/extract_9x9.py --board-size 9 \
  --source 9x9_go/go_raw_dataset/go9 \
  --destination 9x9_go/go_raw_dataset/go9_filtered
python3 go_preprocessing.py --board-size 9 \
  --input-dir 9x9_go/go_raw_dataset/go9_filtered \
  --output-dir 9x9_go/go_dataset_go9 \
  --storage-format npy
python3 train.py --board-size 9 \
  --dataset-dir 9x9_go/go_dataset_go9 \
  --epochs 20
```

### 19×19

For a practical local run, use the professional games archive, filter it to
19×19, and deterministically sample at most 10,000 games for preprocessing:

```bash
cd TRM/
python3 9x9_go/go_raw_dataset/extract_9x9.py --board-size 19 \
  --source 19x19_go/go_raw_dataset/games \
  --destination 19x19_go/go_raw_dataset/19x19_professional_games
python3 go_preprocessing.py --board-size 19 \
  --input-dir 19x19_go/go_raw_dataset/19x19_professional_games \
  --output-dir 19x19_go/go_dataset_professional_10k \
  --max-games 10000 \
  --storage-format npy
python3 train.py --board-size 19 \
  --dataset-dir 19x19_go/go_dataset_professional_10k \
  --epochs 5
```

The 19×19 corpus is much larger, so the 10,000-game sample and NPY format are
recommended for local training. The sample is selected with seed 42. The
previously tested sample produced about 2.1 million examples across the three
splits; counts can vary if the input files change.

For either size, training saves the best inference checkpoint and a resumable
latest checkpoint. The 9×9 checkpoint is `checkpoints/best_model.pt`; 19×19 is
`checkpoints/19x19_go/best_model.pt`. A new 9×9 run writes to the same checkpoint
paths as the existing 9×9 model, so back up those files first if you need to
keep them. Add `--resume` only when continuing a previous run with its
`latest_model.pt`.

The preprocessing script reconstructs missing terminal PASS moves only for
fully processed games with an explicitly scored result. It does not add PASS
moves to games ending by resignation, time loss, or forfeiture.

To evaluate each model on the test split produced by the same pipeline, pass
its path explicitly:

```bash
python3 model_evaluation/go_eval.py --checkpoint checkpoints/best_model.pt \
  --test-data 9x9_go/go_dataset_go9/test
python3 model_evaluation/go_eval.py --checkpoint checkpoints/19x19_go/best_model.pt \
  --test-data 19x19_go/go_dataset_professional_10k/test
```

The inference checkpoint contains only model weights and configuration and is about 3.3 MB of weights, well below GitHub's 100 MB per-file limit. `latest_model.pt` contains optimizer state and remains ignored; the best checkpoint is intentionally allowed by `.gitignore`. After training, add the inference checkpoint to the repository with:
```bash
git add TRM/checkpoints/19x19_go/best_model.pt
git commit -m "Add trained 19x19 Go model"
```
Once that checkpoint is committed, the repository can run 19x19 inference without the datasets. Until then, a fresh clone has no trained 19x19 model.

The policy head produces `N*N + 1` logits: one for each board intersection and one for PASS. The PASS action index is `N*N` (81 on 9x9 and 361 on 19x19). The board size is stored in each new checkpoint and is used automatically by inference and the GUI.

Training saves `best_model.pt` whenever validation loss improves during that training run. The training output's `Train Top-1` is measured on the training split, while `Val Top-1` and the checkpoint's `val_top1` are measured on the validation split. `model_evaluation/go_eval.py` reports accuracy on the separate test split, so its Top-1 can legitimately differ from the checkpoint's validation Top-1. The training and validation metrics are aggregated per example, including a smaller final batch.

Next-move top-1 is imitation accuracy: it asks whether the model predicts the
single move recorded in a human game. It does not say whether that move is
strategically best, whether alternative good moves would be accepted, or
whether a sequence of moves wins a complete game. Game strength is measured
separately by color-balanced matches against fixed opponents under the same
rules.

To **evaluate a trained model** on its corresponding test dataset:
```bash
python3 model_evaluation/go_eval.py --checkpoint checkpoints/best_model.pt
```

Evaluation reads the board size from the checkpoint and automatically chooses
the matching test split. It accepts both `.npz` files and NPY split
directories. To evaluate the 19x19 model:
```bash
python3 model_evaluation/go_eval.py --checkpoint checkpoints/19x19_go/best_model.pt
```

For 9×9, the default evaluator uses `9x9_go/go_dataset_go9/test` (or its
`test.npz` equivalent), matching the preprocessing command documented above.
If a dataset has a different location, pass it explicitly with `--test-data`.

Evaluation metrics, predictions, and plots are saved under
`TRM/model_evaluation/results/` (with a board-size-specific subdirectory for
non-default board sizes).

The older checked-in 9×9 `model_evaluation/results/metrics.json` referred to
`go_dataset/test.npz` and only 2,798 examples, which does not match the current
`go_dataset_go9` split. The checkpoint was re-evaluated on the complete,
documented 41,380-example test split: Top-1 44.74%, Top-3 70.93%, Top-5
82.31%, and PASS precision/recall 61.04%/77.43%. Treat the earlier 2,798-row
metrics as a stale partial evaluation, not as the result for this corpus.
The canonical metrics artifact, predictions, and plots were regenerated
against the full split. The actual 9×9 NPY split sizes are 331,251 train,
41,500 validation, and 41,380 test examples.

## Offline game-strength evaluation

The dedicated `TRM/model_evaluation/` folder now contains the local match
runners, Elo fitter, rules and KataGo resource configuration, downloaded local
KataGo assets, and generated match/evaluation results. The supervised test-set
evaluator is `TRM/model_evaluation/go_eval.py`; the small `TRM/go_eval.py`
wrapper remains for compatibility with older commands. Its default outputs
are stored in `TRM/model_evaluation/results/`.

`model_evaluation/go_match.py` plays either checkpoint against a local KataGo GTP engine. It
uses the checkpoint selected for the requested board size and configures
KataGo with the same rules as `GoGameEnv`: area scoring, 7.5 komi, positional
superko, suicide forbidden, no button or tax, and termination after two
consecutive passes. The full settings for both sizes are in
[`TRM/model_evaluation/config/rules.json`](./TRM/model_evaluation/config/rules.json); the runner checks
KataGo's reported rules and komi before starting. Colors alternate by game.

No Python dependencies are added for this pipeline. For this Linux x86-64
workspace, local KataGo resources are downloaded under the ignored
`TRM/model_evaluation/katago-local/` directory. Their source URLs, licenses, and checksums are
recorded in [`TRM/model_evaluation/config/katago-local-resources.json`](./TRM/model_evaluation/config/katago-local-resources.json).
The runner's `--max-visits` option fixes search effort and thread count so the
opponent strength is repeatable. From `TRM/`, set the local CUDA/cuDNN library
paths:

```bash
export LD_LIBRARY_PATH="$PWD/model_evaluation/katago-local/runtime/nvidia/cudnn/lib:$PWD/model_evaluation/katago-local/engine/squashfs-root/usr/lib:/usr/local/cuda/targets/x86_64-linux/lib:$LD_LIBRARY_PATH"
```

Then run the downloaded 9×9 or 19×19 model:

```bash
python3 model_evaluation/go_match.py --board-size 9 \
  --katago model_evaluation/katago-local/engine/squashfs-root/usr/bin/katago \
  --katago-config model_evaluation/katago-local/engine/default_gtp.cfg \
  --katago-model model_evaluation/katago-local/downloads/kata9x9-b18c384nbt-20231025.bin.gz \
  --max-visits 64 --games 40 \
  --output model_evaluation/results/9x9_go/matches-64-visits.csv

python3 model_evaluation/go_match.py --board-size 19 \
  --katago model_evaluation/katago-local/engine/squashfs-root/usr/bin/katago \
  --katago-config model_evaluation/katago-local/engine/default_gtp.cfg \
  --katago-model model_evaluation/katago-local/downloads/kata1-tf3-b11c768-s12382M-d6533M.bin.gz \
  --max-visits 64 --games 40 \
  --output model_evaluation/results/19x19_go/matches-64-visits.csv
```

The selected 9×9 network is the KataGo finetuned 9×9 model; the 19×19 network
is the latest `kata1` network listed on the official networks page at the time
this setup was prepared (2026-11-07). KataGo reports an internal approximate
rating for its 19×19 model pool, but explicitly warns that those ratings are
not necessarily comparable with ratings from other bots or runs. They are
therefore not KGS/AGA anchors.

For a rough strength bracket, repeat color-balanced series at several search
budgets (the same network and rules, changing only search visits). A screening
ladder can start at 2, 8, 32, and 128 visits; if the TRM's results are all
near 0% or 100%, add lower or higher budgets as needed:

```bash
for visits in 2 8 32 128; do
  python3 model_evaluation/go_match.py --board-size 9 \
    --katago /path/to/katago \
    --katago-config /path/to/gtp.cfg \
    --katago-model /path/to/kata9x9.bin.gz \
    --max-visits "$visits" --games 40 \
    --output "model_evaluation/results/9x9_go/matches-${visits}-visits.csv"
done
```

Compare the three win rates and confidence intervals: this indicates whether
the TRM's strength is below, near, or above those **specific KataGo search
budgets**. Do not combine these runs into a single Elo unless each opponent
tier has an independently calibrated rating on one shared scale.

The runner defaults to `checkpoints/best_model.pt` for 9×9 and
`checkpoints/19x19_go/best_model.pt` for 19×19. It writes one CSV row per
game, including incomplete games at the move limit; incomplete games are
excluded from rating calculations. The output path is overwritten on each
run, so choose a new path to preserve separate batches.

Use the uniform-random legal-move baseline as a sanity check, not as a ranked
Go opponent. It samples uniformly from all legal actions, including PASS, and
alternates colors like the KataGo runner:

```bash
python3 model_evaluation/go_vs_random.py --board-size 9 --games 20 --seed 123 \
  --output model_evaluation/results/9x9_go/random-baseline.csv
python3 model_evaluation/go_elo.py --input model_evaluation/results/9x9_go/random-baseline.csv
```

`model_evaluation/go_vs_random.py` also supports `--board-size 19`; its Elo output is relative
only to this deliberately weak random baseline and must not be compared to
KataGo or external player ratings.

### Early-PASS policy adjustment

The policy has a distinct PASS logit. A diagnostic game showed the 9×9 model
passing repeatedly while the board was still nearly empty. TRM inference now
subtracts 3.5 from the PASS logit while fewer than 16 intersections are
occupied. This applies by default in the GUI, evaluation scripts, and both
match runners; it does not alter checkpoint weights. The setting is exposed
only so controlled ablations can reproduce baseline behavior.

To inspect or recalibrate the PASS-vs-board decision on validation data, use
the diagnostic script. It masks occupied intersections, but does not
reconstruct suicide or positional-superko legality; these are diagnostic
metrics, not a game-strength estimate:

```bash
python3 model_evaluation/go_pass_calibrate.py \
  --checkpoint checkpoints/best_model.pt \
  --validation-data 9x9_go/go_dataset_go9/val \
  --before-stones 16 --penalties 0 1 2 3.5 \
  --output model_evaluation/results/9x9_go/pass-calibration.csv
```

The active setting can be overridden for a controlled ablation by passing
both options to either runner. Zero disables it; both values must be zero
together:

```bash
python3 model_evaluation/go_vs_random.py --board-size 9 --games 40 --seed 123 \
  --pass-logit-penalty 0 --pass-penalty-before-stones 0 \
  --output model_evaluation/results/9x9_go/random-baseline-ablation.csv
```

Match CSVs record the adjustment and count PASS decisions, including those
made with fewer than 16 stones on the board. Keep baseline and adjusted runs
in separate files, balance colors, and do not mix their scores into one Elo
fit.

An initial 9×9 validation sweep on 41,500 examples found that a 3.5-logit
penalty below 16 occupied points changed PASS precision/recall from
63.54%/80.40% to 64.39%/78.70%; the PASS F1 was nearly unchanged (70.98% vs
70.83%). The diagnostic masks occupied intersections, but not suicide or
superko, so this is not evidence that the game-playing policy is stronger.
In a separate 20-game-per-condition screening run with the same seed and
independent opponent RNG streams, the baseline went 15-5 against uniform
random (all five losses as White) and made 76 PASS moves before 16 points were
occupied. The adjusted version went 20-0 and made no such early PASS moves.
Against KataGo at one visit, it went from 0-20 to 1-19. These small samples
support using the adjustment to avoid the observed early-PASS failure mode,
but do not establish a general strength gain. In particular, it slightly
reduced validation PASS recall and did not materially improve results against
KataGo. The comparison CSVs and validation calibration table are under
`TRM/model_evaluation/results/diagnostic-2026-11-07/`.

The earlier KataGo ladder under
`TRM/model_evaluation/results/katago-benchmark-2026-11-07/` predates the default
adjustment and therefore measures the unadjusted checkpoint policy. New
matches use the adjusted policy unless both options are explicitly set to
zero.

### Weaker, human-style KataGo opponents

The previously tested 1-visit opponent was still the strong KataGo network;
reducing search visits further is not a useful human-rank ladder. KataGo also
publishes a supervised HumanSL network whose configurable profiles imitate
human moves at ranks from 20 kyu upward. The network is downloaded locally
under `model_evaluation/katago-local/downloads/` and its URL, SHA-256, and license are recorded
in `model_evaluation/config/katago-local-resources.json`.

Use the standard KataGo network for the selected board size as the main
network, add the HumanSL network, and select profiles in separate runs:

```bash
for profile in rank_20k rank_15k rank_10k rank_5k rank_1k; do
  python3 model_evaluation/go_match.py --board-size 9 \
    --katago model_evaluation/katago-local/engine/squashfs-root/usr/bin/katago \
    --katago-config model_evaluation/katago-local/engine/default_gtp.cfg \
    --katago-model model_evaluation/katago-local/downloads/kata9x9-b18c384nbt-20231025.bin.gz \
    --katago-human-model model_evaluation/katago-local/downloads/b18c384nbt-humanv0.bin.gz \
    --human-profile "$profile" --max-visits 40 --games 40 \
    --output "model_evaluation/results/9x9_go/human-${profile}.csv"
done
```

For 19×19, use its 19×19 main network and the same HumanSL profile network:

```bash
for profile in rank_20k rank_15k rank_10k rank_5k rank_1k; do
  python3 model_evaluation/go_match.py --board-size 19 \
    --katago model_evaluation/katago-local/engine/squashfs-root/usr/bin/katago \
    --katago-config model_evaluation/katago-local/engine/default_gtp.cfg \
    --katago-model model_evaluation/katago-local/downloads/kata1-tf3-b11c768-s12382M-d6533M.bin.gz \
    --katago-human-model model_evaluation/katago-local/downloads/b18c384nbt-humanv0.bin.gz \
    --human-profile "$profile" --max-visits 40 --games 40 \
    --output "model_evaluation/results/19x19_go/human-${profile}.csv"
done
```

`--human-profile` supports `rank_20k`, `rank_15k`, `rank_10k`, `rank_5k`,
`rank_1k`, `rank_1d`, `rank_3d`, `rank_5d`, and `rank_9d`. If the option is
omitted with `--katago-human-model`, it defaults to `rank_20k`. KataGo's
HumanSL profiles are human-move imitation settings, not exact measured Elo
anchors; their actual win rates against the TRM must be tested. Start at the
weakest profiles and move upward. When a profile is near 50%, run a larger
color-balanced sample (for example 100 games) and report its confidence
interval. If 20/20 wins or losses recur, bracket between profiles rather than
claiming a precise rating. These profile results are relative game outcomes,
not official KGS/AGA conversions.

To sample the broad human-policy distribution while using one KataGo visit,
pass `--human-temperature-mode full --max-visits 1`. This uses the
temperature settings from KataGo v1.18.2's
[`gtp_human5k_example.cfg`](https://github.com/lightvector/KataGo/blob/v1.18.2/cpp/configs/gtp_human5k_example.cfg):
early temperature 0.85, later temperature 0.70, an 80-move half-life, and
temperature applied only below 1% move probability. Here “full” means
KataGo's documented broad human-style sampling configuration, not a literal
temperature value of 1.0. The mode is written to each match CSV.

#### Initial local results

The first 100-game, color-balanced runs against `rank_20k` used 40 visits and
the default early-PASS adjustment. All games completed under the configured
custom rules:

| Board | TRM record | Win rate (95% Wilson interval) | Relative Elo vs this fixed profile (95% CI) |
|---|---:|---:|---:|
| 9×9 | 35–65 | 35% (26.4%–44.7%) | −108 (−179 to −36) |
| 19×19 | 54–46 | 54% (44.3%–63.4%) | +28 (−40 to +96) |

The 19×19 result is consistent with an approximately even match. For 9×9,
`rank_20k` is the weakest published HumanSL profile, but the TRM won only
35/100; higher profiles were harder in a 20-game screen (rank_15k: 4/20,
rank_10k: 3/20, rank_5k: 1/20, rank_1k: 0/20; rank_20k: 6/20). That screen is
small and exploratory. There is no weaker published HumanSL profile to test.
If a closer 9×9 50% match is needed, a deliberately randomized or mixed
opponent would be a custom calibration tier, not a KGS/AGA rank or Elo anchor.

The detailed records are
`TRM/model_evaluation/results/human-ladder-2026-11-07/9x9-rank_20k-100.csv`,
`TRM/model_evaluation/results/human-ladder-2026-11-07/19x19-rank_20k-100.csv`, and
the 9×9 profile-screen CSVs in the same folder. The corresponding relative-Elo
fits are saved as `9x9-rank_20k-100-elo.json` and
`19x19-rank_20k-100-elo.json`. Since the HumanSL profile has no measured Elo
anchor, the reported differences are only fitted comparisons against this
exact local opponent configuration; they are not absolute Elo values or
official Go ranks.

#### One-visit, full-temperature experiment

The following screening experiment used one KataGo visit, the full-range
HumanSL temperature settings described above, the default TRM PASS adjustment,
and 20 color-balanced games per profile. `rank_20k` was then run for 100 games
per board size. Counts are TRM wins; the remaining games were losses (there
were no draws):

| Board | `rank_20k` (100 games) | `rank_15k` | `rank_10k` | `rank_5k` | `rank_1k` |
|---|---:|---:|---:|---:|---:|
| 9×9 | 39/100 (39%) | 4/20 (20%) | 3/20 (15%) | 2/20 (10%) | 1/20 (5%) |
| 19×19 | 25/100 (25%) | 1/20 (5%) | 3/20 (15%) | 0/20 (0%) | 0/20 (0%) |

The 95% Wilson intervals for the 100-game `rank_20k` results are 30.0%–48.8%
on 9×9 and 17.5%–34.3% on 19×19. Relative Elo fits against these exact
configurations are −78 (95% CI −148 to −8) and −191 (95% CI −269 to −112),
respectively. These are relative estimates, not absolute ratings or rank
conversions.

Compared with the earlier 40-visit runs, the 9×9 result is broadly similar
(35/100 previously versus 39/100 here). The 19×19 result is lower (54/100
previously versus 25/100 here). This is not a pure visit-count comparison:
the earlier runs did not explicitly set the full HumanSL temperature mode.
KataGo's v1.18.2 HumanSL example config says that with
`humanSLChosenMoveProp=1.0` and a very large
`humanSLChosenMovePiklLambda`, increasing visits does not affect move
selection except potentially pass/resign judgment. Thus this result is more
consistent with the changed sampling/temperature behavior (and possibly pass
judgment) than with search visits directly making the opponent stronger.
The profile screening is exploratory, especially at 20 games per tier; only
the `rank_20k` outcomes have the larger 100-game sample.

The CSVs and relative-Elo JSONs are in
`TRM/model_evaluation/results/human-onevisit-2026-11-08/`. To reproduce the settings,
use `--max-visits 1 --human-temperature-mode full`; the runner records both
the visit budget and temperature mode in every match row.

#### Requested 40-game profile ladder

The follow-up ladder ran exactly 40 games for every profile and board size,
with 20 games as each color, one visit, and the same full-range HumanSL
temperature settings. The tables count results from the TRM's perspective;
all 400 games completed and there were no draws.

| Board | HumanSL profile | TRM wins as White | TRM wins as Black | TRM losses as White | TRM losses as Black |
|---|---|---:|---:|---:|---:|
| 9×9 | rank_1k | 5 | 0 | 15 | 20 |
| 9×9 | rank_5k | 5 | 3 | 15 | 17 |
| 9×9 | rank_10k | 7 | 1 | 13 | 19 |
| 9×9 | rank_15k | 13 | 2 | 7 | 18 |
| 9×9 | rank_20k | 15 | 5 | 5 | 15 |
| 19×19 | rank_1k | 0 | 0 | 20 | 20 |
| 19×19 | rank_5k | 0 | 1 | 20 | 19 |
| 19×19 | rank_10k | 3 | 2 | 17 | 18 |
| 19×19 | rank_15k | 1 | 1 | 19 | 19 |
| 19×19 | rank_20k | 8 | 4 | 12 | 16 |

At 9×9, the TRM split 20–20 against `rank_20k`, but most wins came as White
(15/20). At 19×19, it won 12/40 against `rank_20k` and none against `rank_1k`.
The 19×19 intermediate profile results are not monotonic (`rank_10k` exceeded
`rank_15k` in this sample), so do not read small rank-to-rank differences as
a precise strength ordering. HumanSL labels remain imitation profiles, not
official calibrated ranks. Confidence intervals and full protocol notes are
in [`TRM/model_evaluation/reports/humansl-onevisit-40-2026-11-08.md`](./TRM/model_evaluation/reports/humansl-onevisit-40-2026-11-08.md);
the raw per-game CSVs and summary are under
`TRM/model_evaluation/results/human-profile-40-2026-11-08/`.

The profile ladder can be reproduced from `TRM/` with the following
board-specific commands; each invocation alternates TRM color, yielding 20
games as Black and 20 as White:

```bash
for profile in rank_1k rank_5k rank_10k rank_15k rank_20k; do
  python3 model_evaluation/go_match.py --board-size 9 \
    --katago model_evaluation/katago-local/engine/squashfs-root/usr/bin/katago \
    --katago-config model_evaluation/katago-local/engine/default_gtp.cfg \
    --katago-model model_evaluation/katago-local/downloads/kata9x9-b18c384nbt-20231025.bin.gz \
    --katago-human-model model_evaluation/katago-local/downloads/b18c384nbt-humanv0.bin.gz \
    --human-profile "$profile" --human-temperature-mode full \
    --max-visits 1 --games 40 \
    --output "model_evaluation/results/human-profile-40-2026-11-08/9x9-${profile}-40.csv"
done
```

For 19×19, use the same command with `--board-size 19` and
`--katago-model model_evaluation/katago-local/downloads/kata1-tf3-b11c768-s12382M-d6533M.bin.gz`,
and change the output filename prefix to `19x19`.

Fit the target's Elo difference against the exact KataGo model/configuration:

```bash
python3 model_evaluation/go_elo.py --input model_evaluation/results/9x9_go/matches.csv
```

Each summary reports the maximum-likelihood rating difference on the standard
400-point Elo scale (wins = 1, draws = 0.5), the outcome counts, and an
approximate 95% confidence interval. An all-win or all-loss sample has an
unbounded Elo maximum-likelihood estimate and is reported as such instead of
an arbitrary finite rating. Use `--anchor-elo` only when the exact reference
checkpoint and its KataGo search settings have a known rating on the scale you
intend to report; otherwise the result is relative Elo, not a transferable
absolute rating.

The local KataGo download is ignored by Git and can be recreated from the
official [KataGo v1.18.2 release](https://github.com/lightvector/KataGo/releases/tag/v1.18.2),
the [official 9×9 network page](https://katagotraining.org/extra_networks/),
and the [official KataGo networks page](https://katagotraining.org/networks/).
The selected network files have their own terms listed in
[KataGo's network license](https://katagotraining.org/network_license/).
For this NVIDIA CUDA setup, the extracted KataGo binary also needs NVIDIA's
cuDNN runtime. CUDA/cuDNN are free to use locally but are proprietary, not
open-source; if the requirement is that every runtime component be open
source, use KataGo's OpenCL or CPU build instead.

KataGo strength depends on its weights, search settings, and hardware/time
budget. A KataGo rating is not automatically a KGS or AGA rating, and there is
no universal rank conversion from a single match series. To print a KGS/AGA
rank tier, supply a locally prepared and documented threshold file with
`rank,elo_min` columns via `--rank-tiers-csv` together with `--anchor-elo`.
Each `elo_min` must be a measured/calibrated lower Elo boundary for that rank
on the same scale; no unverified rank mapping is bundled. The same procedure
and custom rules apply to both board sizes, but 9×9 and 19×19 ratings should
be reported separately.

The optional micro-training check uses the selected board size and loads either an
NPZ or NPY train split when available; if no dataset exists it uses synthetic
data:
```bash
python3 micro_train.py --board-size 9
python3 micro_train.py --board-size 19
```

# Dataset provenance and shapes

- **9×9:** source `Go Quest – Top 50 players`. The current raw corpus at
  `TRM/9x9_go/go_raw_dataset/go9/` contains 8,606 SGF files. The extraction
  step retains the readable 9×9 games and removes exact duplicates.
- **19×19:** source `CWI Database of Go Games`
  (https://homepages.cwi.nl/~aeb/go/games/). The practical training pipeline
  deliberately uses the professional `games/` archive only, not the complete
  archive or the additional `jgdb/` source. The current `games/` folder has
  97,565 SGF files; size filtering and exact deduplication produce 95,715
  unique 19×19 files. Preprocessing then deterministically samples at most
  10,000 games (seed 42) for local training.

Raw SGF games have variable numbers of moves, so they do not have one fixed
tensor shape. The processed arrays use `X: [examples, 7, N, N]` (`float32`)
and `y: [examples]` (`int64`). The current/previously verified NPY splits
have these shapes:

| Board size | Raw SGFs selected for the pipeline | Train examples | Validation examples | Test examples | Total examples |
|---|---:|---:|---:|---:|---:|
| 9×9 | 8,606 | 331,251 | 41,500 | 41,380 | 414,131 |
| 19×19 | 95,715 unique; at most 10,000 sampled | 1,668,062 | 208,594 | 208,653 | 2,085,309 |

These are counts from the datasets currently generated and can change if the
input SGFs or sampled games change. NPY split directories contain `X.npy` and
`y.npy` and are excluded from Git along with the raw archives. The larger
19×19 full archive is available separately; it is not the recommended input
for the local training run described above.

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
TRM/model_evaluation/go_eval.py
```
This file evaluates supervised next-move prediction on the test split
associated with the selected checkpoint. Board size is read from the
checkpoint; the evaluator accepts NPZ files and memory-mapped NPY split
directories. It saves metrics, predictions, and diagnostic plots under the
size-specific `model_evaluation/results/` folder. Test Top-1 is not expected to
match validation Top-1 exactly.

```bash
TRM/tests/test_go_preprocessing.py
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
`TRM/tests/test_game_env.py` checks board rules and legal-move behavior.
`TRM/tests/test_go_board_size.py` checks model output dimensions, GUI
checkpoint selection, board symmetries, and action-selection helpers for both
board sizes. `TRM/tests/test_go_preprocessing.py` checks SGF
filtering/deduplication, position counting, NPZ/NPY dataset writing, and
dataset loading.

Run all unit tests from the repository root with:
```bash
python3 -m pytest
```

# Pipeline
PIPELINE FOR TRAINING PHASE:
![Pipeline per training](/TRM/assets/pipeline_for_trainig.png)

PIPELINE FOR INFERENCE PHASE:
![Pipeline for inference](/TRM/assets/pipeline_inference.png)
