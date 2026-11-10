# Experiments

This page records inference choices and exploratory results. Match results
are specific to their opponent configuration and protocol; they are not
general Go ratings.

## Near-best move selection

The agent samples uniformly from legal actions whose probability is at least
`best_probability * (1 - tolerance)`. With the default `tolerance=0.02`, the
cutoff is 98% of the best legal probability. This is a relative tolerance:
it is not a two-percentage-point absolute difference. If only the best action
meets the cutoff, that action is selected. The same selection rule is used
for greedy self-play in `go_pass_eval.py`.

## Early-PASS adjustment

A diagnostic showed repeated PASS moves early in 9×9 games. By default,
inference subtracts 3.5 from the PASS logit while fewer than 16 intersections
are occupied. This setting does not alter checkpoint weights and is applied
in the GUI and evaluation/match runners. Set both
`--pass-logit-penalty 0` and `--pass-penalty-before-stones 0` to disable it for
a controlled ablation.

On the 41,500-example 9×9 validation split, the adjustment changed PASS
precision/recall from 63.54%/80.40% to 64.39%/78.70%; PASS F1 was nearly
unchanged (70.98% versus 70.83%). The diagnostic does not reproduce all game
legality constraints, so this is not evidence of stronger play.

## Recorded KataGo HumanSL matches

The 100-game, color-balanced `rank_20k` runs at 40 visits recorded:

| Board | TRM record | Win rate (95% Wilson interval) | Relative Elo vs this configuration (95% CI) |
|---|---:|---:|---:|
| 9×9 | 35–65 | 35% (26.4%–44.7%) | −108 (−179 to −36) |
| 19×19 | 54–46 | 54% (44.3%–63.4%) | +28 (−40 to +96) |

The follow-up one-visit, full-HumanSL-temperature 100-game `rank_20k` runs
recorded 39/100 wins on 9×9 and 25/100 on 19×19. This is not a pure visit
count comparison: the temperature/sampling configuration also changed. The
relative Elo fits were −78 (95% CI −148 to −8) and −191 (95% CI −269 to −112).

The later one-visit profile ladder ran 40 games per profile and board size,
with 20 games per color. Against `rank_20k`, the TRM scored 20/40 on 9×9 and
12/40 on 19×19; against `rank_1k`, it scored 5/40 and 0/40 respectively.
Intermediate profile results were not monotonic in every small sample.
HumanSL labels are imitation profiles and do not provide calibrated rank
anchors. Confidence intervals and per-game records are in
`TRM/model_evaluation/reports/` and `TRM/model_evaluation/results/`.

For reproducible comparisons, retain the board size, KataGo network, rules,
visit limit, profile, temperature mode, seed, and color balance with each
result. Do not combine experiments with different settings into a single
rating fit.
