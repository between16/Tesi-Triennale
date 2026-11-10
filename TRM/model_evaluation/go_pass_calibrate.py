"""Measure an optional early-game PASS-logit adjustment on validation data.

The board candidate mask excludes occupied intersections but does not model
suicide or positional-superko legality. Results are a supervised diagnostic,
not a substitute for full-game evaluation.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
TRM_DIR = SCRIPT_DIR.parent
if str(TRM_DIR) not in sys.path:
    sys.path.insert(0, str(TRM_DIR))

from go_dataset import GoDataset, default_go_dataset_dir
from models.go_model import GoTRMModel


def pass_decision_metrics(
    pass_margins: np.ndarray,
    occupied_counts: np.ndarray,
    targets: np.ndarray,
    pass_action: int,
    penalty: float,
    before_stones: int,
) -> dict[str, float | int]:
    """Score PASS-vs-board decisions after an optional early-game adjustment."""
    if not math.isfinite(penalty) or penalty < 0:
        raise ValueError("penalty must be finite and non-negative.")
    if isinstance(before_stones, bool) or not isinstance(before_stones, int):
        raise ValueError("before_stones must be a non-negative integer.")
    if before_stones < 0 or ((penalty == 0) != (before_stones == 0)):
        raise ValueError("Disable both settings with zero or set both to positive values.")
    if not (
        pass_margins.ndim == occupied_counts.ndim == targets.ndim == 1
        and len(pass_margins) == len(occupied_counts) == len(targets)
        and len(targets) > 0
    ):
        raise ValueError("Margins, stone counts, and targets must be aligned non-empty arrays.")

    adjusted_margins = pass_margins.copy()
    if penalty:
        adjusted_margins[occupied_counts < before_stones] -= penalty
    predicted_pass = adjusted_margins > 0
    if pass_action < 1:
        raise ValueError("pass_action must be positive.")
    actual_pass = targets == pass_action

    true_positive = int(np.sum(predicted_pass & actual_pass))
    false_positive = int(np.sum(predicted_pass & ~actual_pass))
    false_negative = int(np.sum(~predicted_pass & actual_pass))
    true_negative = int(np.sum(~predicted_pass & ~actual_pass))
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (true_positive + true_negative) / len(targets)
    return {
        "examples": len(targets),
        "pass_examples": int(np.sum(actual_pass)),
        "pass_precision": precision,
        "pass_recall": recall,
        "pass_f1": f1,
        "pass_binary_accuracy": accuracy,
        "false_passes": false_positive,
        "missed_passes": false_negative,
        "correct_passes": true_positive,
    }


def default_validation_data_path(board_size: int) -> Path:
    """Locate the repository's conventional validation split."""
    board_dir = TRM_DIR / f"{board_size}x{board_size}_go"
    dataset_dir = default_go_dataset_dir(board_size, TRM_DIR)
    npy_validation = dataset_dir / "val"
    if npy_validation.is_dir():
        return npy_validation
    npz_validation = dataset_dir / "val.npz"
    if npz_validation.is_file():
        return npz_validation
    legacy_dir = board_dir / "go_dataset"
    legacy_npy = legacy_dir / "val"
    if legacy_npy.is_dir():
        return legacy_npy
    legacy_npz = legacy_dir / "val.npz"
    return legacy_npz if legacy_npz.is_file() else npy_validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure early-game PASS-logit adjustments on supervised validation data."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--before-stones", type=int, default=16)
    parser.add_argument(
        "--penalties",
        type=float,
        nargs="+",
        default=[0.0, 1.0, 2.0, 3.5],
        help="PASS-logit penalties to compare; zero is the unmodified baseline.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    if args.before_stones < 1:
        raise ValueError("--before-stones must be positive.")
    if not args.penalties or any(
        not math.isfinite(value) or value < 0 for value in args.penalties
    ):
        raise ValueError("--penalties must contain finite non-negative values.")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if "config_dict" not in checkpoint or "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint must contain 'config_dict' and 'model_state_dict'.")
    board_size = int(checkpoint["config_dict"].get("board_size", 9))
    validation_data = args.validation_data or default_validation_data_path(board_size)
    if not validation_data.is_file() and not validation_data.is_dir():
        raise FileNotFoundError(f"Validation dataset not found: {validation_data}")

    dataset = GoDataset(validation_data, augment=False, board_size=board_size)
    if len(dataset) == 0:
        raise ValueError("Validation dataset is empty.")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = GoTRMModel(checkpoint["config_dict"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    pass_margins: list[np.ndarray] = []
    occupied_counts: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    pass_action = board_size * board_size
    with torch.inference_mode():
        for features, labels in loader:
            features = features.to(device, non_blocking=device.type == "cuda")
            logits = model(features).float()
            occupied = (features[:, 0] + features[:, 1]).gt(0).flatten(1)
            board_logits = logits[:, :pass_action].masked_fill(occupied, float("-inf"))
            best_board_logits = board_logits.max(dim=1).values
            pass_margins.append((logits[:, pass_action] - best_board_logits).cpu().numpy())
            occupied_counts.append(occupied.sum(dim=1).cpu().numpy())
            targets.append(labels.numpy())

    margins = np.concatenate(pass_margins)
    stones = np.concatenate(occupied_counts)
    labels = np.concatenate(targets)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "board_size",
        "checkpoint",
        "validation_data",
        "validation_examples",
        "penalty_logit",
        "before_stones",
        *pass_decision_metrics(
            margins,
            stones,
            labels,
            pass_action=pass_action,
            penalty=0.0,
            before_stones=0,
        ).keys(),
        "legality_note",
    ]
    with args.output.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for penalty in args.penalties:
            gate = 0 if penalty == 0 else args.before_stones
            report = pass_decision_metrics(
                margins,
                stones,
                labels,
                pass_action=pass_action,
                penalty=penalty,
                before_stones=gate,
            )
            writer.writerow(
                {
                    "board_size": board_size,
                    "checkpoint": args.checkpoint,
                    "validation_data": validation_data,
                    "validation_examples": len(dataset),
                    "penalty_logit": penalty,
                    "before_stones": gate,
                    **report,
                    "legality_note": "occupied intersections masked; suicide and superko not modeled",
                }
            )
    print(f"PASS calibration results: {args.output}")


if __name__ == "__main__":
    main()
