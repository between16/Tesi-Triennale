"""Fit an Elo estimate from completed TRM-vs-reference match results.

The score-based likelihood uses the standard Elo expected-score curve. A draw
contributes half a point. Rank conversion is deliberately opt-in: the supplied
CSV must contain calibrated external-scale thresholds, not guessed mappings.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

ELO_SCALE = 400.0
CONFIDENCE_Z = 1.96


def fit_elo(results: list[dict[str, str]], board_size: int | None = None) -> dict[str, Any]:
    """Return the maximum-likelihood Elo difference from completed outcomes."""
    completed = [
        row
        for row in results
        if row.get("completed", "").lower() == "true"
    ]
    if not completed:
        raise ValueError("No completed match results are available.")

    sizes = {int(row["board_size"]) for row in completed}
    if board_size is None:
        if len(sizes) != 1:
            raise ValueError(
                "Match results contain multiple board sizes; select one with --board-size."
            )
        board_size = next(iter(sizes))
    selected = [row for row in completed if int(row["board_size"]) == board_size]
    if not selected:
        raise ValueError(f"No completed match results are available for {board_size}x{board_size}.")

    configuration: dict[str, str | None] = {}
    for field in (
        "target_checkpoint",
        "katago_model",
        "katago_config",
        "search_max_visits",
        "human_model",
        "opponent_profile",
        "human_temperature_mode",
        "opponent_type",
        "search_mode",
        "search_depth",
        "search_candidates",
        "pass_logit_penalty",
        "pass_penalty_before_stones",
        "komi",
    ):
        if field in {"pass_logit_penalty", "pass_penalty_before_stones"}:
            values = {row.get(field) or "0" for row in selected}
        else:
            values = {row[field] for row in selected if row.get(field)}
        if len(values) > 1:
            raise ValueError(
                f"Match results mix multiple {field} values; fit each configuration separately."
            )
        configuration[field] = next(iter(values)) if values else None

    scores = {"win": 1.0, "draw": 0.5, "loss": 0.0}
    try:
        observations = [scores[row["target_result"].lower()] for row in selected]
    except KeyError as exc:
        raise ValueError(f"Unknown or missing target_result: {exc.args[0]!r}.") from exc
    if not observations:
        raise ValueError("No completed match results are available for this board size.")

    games = len(observations)
    score_sum = sum(observations)
    score_rate = score_sum / games
    estimate: float | None
    standard_error: float | None
    if score_rate == 0.0:
        estimate = None
        standard_error = None
        status = "unbounded_below"
    elif score_rate == 1.0:
        estimate = None
        standard_error = None
        status = "unbounded_above"
    else:
        estimate = ELO_SCALE * math.log10(score_rate / (1.0 - score_rate))
        standard_error = (
            ELO_SCALE
            / math.log(10.0)
            * math.sqrt(1.0 / (games * score_rate * (1.0 - score_rate)))
        )
        status = "finite_mle"

    wins = sum(value == 1.0 for value in observations)
    draws = sum(value == 0.5 for value in observations)
    losses = sum(value == 0.0 for value in observations)
    return {
        "board_size": board_size,
        "match_configuration": configuration,
        "games": games,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score_rate": score_rate,
        "elo_difference": estimate,
        "elo_difference_95ci": (
            None
            if estimate is None or standard_error is None
            else [
                estimate - CONFIDENCE_Z * standard_error,
                estimate + CONFIDENCE_Z * standard_error,
            ]
        ),
        "standard_error": standard_error,
        "rating_status": status,
        "estimator": "maximum-likelihood score fit to the standard Elo curve; draws score 0.5",
    }


def load_rank_thresholds(path: Path) -> list[tuple[float, str]]:
    """Load externally calibrated rank thresholds from a rank,elo_min CSV."""
    thresholds: list[tuple[float, str]] = []
    with path.open(newline="", encoding="utf-8") as calibration_file:
        reader = csv.DictReader(calibration_file)
        if reader.fieldnames is None or not {"rank", "elo_min"}.issubset(reader.fieldnames):
            raise ValueError("Rank calibration CSV must contain rank and elo_min columns.")
        for row in reader:
            rank = row["rank"].strip()
            try:
                elo_min = float(row["elo_min"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid elo_min in rank calibration row: {row}.") from exc
            if not rank or not math.isfinite(elo_min):
                raise ValueError(f"Invalid rank calibration row: {row}.")
            thresholds.append((elo_min, rank))
    if not thresholds:
        raise ValueError("Rank calibration CSV contains no tiers.")
    if len({elo for elo, _ in thresholds}) != len(thresholds):
        raise ValueError("Rank calibration elo_min values must be unique.")
    return sorted(thresholds)


def rank_for_rating(rating: float, thresholds: list[tuple[float, str]]) -> str:
    eligible = [(elo, rank) for elo, rank in thresholds if elo <= rating]
    if not eligible:
        return f"below {thresholds[0][1]}"
    return max(eligible)[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate relative Elo from a go_match.py results CSV."
    )
    parser.add_argument("--input", type=Path, required=True, help="Match results CSV.")
    parser.add_argument("--output", type=Path, default=None, help="JSON summary output.")
    parser.add_argument("--board-size", type=int, choices=(9, 19), default=None)
    parser.add_argument(
        "--anchor-elo",
        type=float,
        default=None,
        help="Known numeric Elo for the exact reference checkpoint/configuration.",
    )
    parser.add_argument(
        "--rank-tiers-csv",
        type=Path,
        default=None,
        help="Optional calibrated CSV with rank,elo_min columns.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Match results CSV not found: {args.input}")
    if args.anchor_elo is not None and not math.isfinite(args.anchor_elo):
        raise ValueError("--anchor-elo must be finite.")
    if args.rank_tiers_csv is not None and args.anchor_elo is None:
        raise ValueError("--rank-tiers-csv requires --anchor-elo.")

    with args.input.open(newline="", encoding="utf-8") as results_file:
        report = fit_elo(list(csv.DictReader(results_file)), args.board_size)

    report["anchor_elo"] = args.anchor_elo
    report["absolute_elo"] = (
        None
        if args.anchor_elo is None or report["elo_difference"] is None
        else args.anchor_elo + report["elo_difference"]
    )
    report["rank_tier"] = None
    if args.rank_tiers_csv is not None and report["absolute_elo"] is not None:
        report["rank_tier"] = rank_for_rating(
            report["absolute_elo"], load_rank_thresholds(args.rank_tiers_csv)
        )
    report["rank_calibration_file"] = (
        str(args.rank_tiers_csv) if args.rank_tiers_csv is not None else None
    )

    output = args.output or args.input.with_suffix(".elo.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))
    print(f"Elo summary: {output}")


if __name__ == "__main__":
    main()
