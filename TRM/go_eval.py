"""Evaluate a trained GoTRM checkpoint on the held-out test split.
Run from any working directory:
    python /path/to/TRM/evaluate_model.py
    python /path/to/TRM/evaluate_model.py --checkpoint /path/to/best_model.pt
This evaluates supervised next-move prediction. It does not measure game-playing
strength or win rate.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Save figures without requiring a graphical session.
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Make local TRM modules importable regardless of the current working directory.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from agents import GoTRMModel
from go_dataset import GoDataset

BOARD_SIZE = 9
PASS_MOVE = BOARD_SIZE * BOARD_SIZE
NUM_ACTIONS = PASS_MOVE + 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained GoTRM checkpoint on the held-out 9x9 test set."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=SCRIPT_DIR / "checkpoints" / "best_model.pt",
        help="Checkpoint produced by train.py (default: TRM/checkpoints/best_model.pt).",
    )
    parser.add_argument(
        "--test-data",
        type=Path,
        default=SCRIPT_DIR / "9x9_go" / "go_dataset" / "test.npz",
        help="Test .npz containing X and y.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "evaluation_results",
        help="Directory for metrics, predictions, and plots.",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device to use; auto selects CUDA when available.",
    )
    return parser.parse_args()


def safe_ratio(numerator: int, denominator: int) -> float | None:
    """Return None when a metric has no valid denominator."""
    if denominator == 0:
        return None
    return float(numerator / denominator)


def action_name(action: int) -> str:
    """Convert a flat action index into a readable board coordinate or PASS."""
    if action == PASS_MOVE:
        return "PASS"
    row, col = divmod(int(action), BOARD_SIZE)
    return f"row_{row + 1}_col_{col + 1}"


def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict, dict[str, np.ndarray]]:
    model.eval()

    sample_count = 0
    loss_sum = 0.0
    topk_correct = {1: 0, 3: 0, 5: 0}

    all_targets: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []
    all_confidences: list[np.ndarray] = []
    all_true_probabilities: list[np.ndarray] = []
    all_pass_probabilities: list[np.ndarray] = []
    all_topk: list[np.ndarray] = []

    with torch.inference_mode():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device, non_blocking=(device.type == "cuda"))
            y_batch = y_batch.to(device, non_blocking=(device.type == "cuda"))

            logits = model(x_batch).float()
            if logits.ndim != 2 or logits.shape[1] != NUM_ACTIONS:
                raise ValueError(
                    f"Expected model output [batch, {NUM_ACTIONS}], "
                    f"received {tuple(logits.shape)}."
                )
            if torch.any(y_batch < 0) or torch.any(y_batch >= NUM_ACTIONS):
                raise ValueError("Test labels must be in the range 0..81 (81 means PASS).")

            batch_size = int(y_batch.shape[0])
            probabilities = torch.softmax(logits, dim=-1)
            loss_sum += F.cross_entropy(logits, y_batch, reduction="sum").item()
            sample_count += batch_size

            topk = torch.topk(logits, k=5, dim=-1).indices
            for k in topk_correct:
                topk_correct[k] += int((topk[:, :k] == y_batch[:, None]).any(dim=1).sum().item())

            all_targets.append(y_batch.cpu().numpy())
            all_predictions.append(topk[:, 0].cpu().numpy())
            all_topk.append(topk.cpu().numpy())
            all_confidences.append(probabilities.max(dim=-1).values.cpu().numpy())
            all_true_probabilities.append(
                probabilities.gather(1, y_batch[:, None]).squeeze(1).cpu().numpy()
            )
            all_pass_probabilities.append(probabilities[:, PASS_MOVE].cpu().numpy())

    if sample_count == 0:
        raise ValueError("The test dataset is empty; evaluation cannot continue.")

    targets = np.concatenate(all_targets).astype(np.int64)
    predictions = np.concatenate(all_predictions).astype(np.int64)
    topk_predictions = np.concatenate(all_topk).astype(np.int64)
    confidence = np.concatenate(all_confidences)
    true_action_probability = np.concatenate(all_true_probabilities)
    pass_probability = np.concatenate(all_pass_probabilities)

    true_is_pass = targets == PASS_MOVE
    predicted_is_pass = predictions == PASS_MOVE

    tp = int(np.sum(true_is_pass & predicted_is_pass))
    fp = int(np.sum(~true_is_pass & predicted_is_pass))
    fn = int(np.sum(true_is_pass & ~predicted_is_pass))
    tn = int(np.sum(~true_is_pass & ~predicted_is_pass))

    pass_precision = safe_ratio(tp, tp + fp)
    pass_recall = safe_ratio(tp, tp + fn)
    if pass_precision is None or pass_recall is None or (pass_precision + pass_recall) == 0:
        pass_f1 = None
    else:
        pass_f1 = float(2 * pass_precision * pass_recall / (pass_precision + pass_recall))

    board_targets = ~true_is_pass
    board_top1_accuracy = safe_ratio(
        int(np.sum((predictions == targets) & board_targets)),
        int(np.sum(board_targets)),
    )

    metrics = {
        "checkpoint": None,
        "test_data": None,
        "num_examples": int(sample_count),
        "cross_entropy_loss": float(loss_sum / sample_count),
        "top1_accuracy": float(topk_correct[1] / sample_count),
        "top3_accuracy": float(topk_correct[3] / sample_count),
        "top5_accuracy": float(topk_correct[5] / sample_count),
        "board_move_examples": int(np.sum(board_targets)),
        "pass_examples": int(np.sum(true_is_pass)),
        "board_move_top1_accuracy": board_top1_accuracy,
        "pass_precision": pass_precision,
        "pass_recall": pass_recall,
        "pass_f1": pass_f1,
        "pass_true_positives": tp,
        "pass_false_positives": fp,
        "pass_false_negatives": fn,
        "pass_true_negatives": tn,
        "mean_probability_of_correct_action": float(np.mean(true_action_probability)),
        "mean_pass_probability_when_target_is_board_move": (
            float(np.mean(pass_probability[board_targets])) if np.any(board_targets) else None
        ),
        "mean_pass_probability_when_target_is_pass": (
            float(np.mean(pass_probability[true_is_pass])) if np.any(true_is_pass) else None
        ),
        "pass_probability_brier_score": float(
            np.mean((pass_probability - true_is_pass.astype(np.float64)) ** 2)
        ),
        "warning": (
            "The test split contains fewer than 20 PASS targets; PASS-specific metrics "
            "are highly uncertain and should be treated as diagnostic only."
            if int(np.sum(true_is_pass)) < 20
            else None
        ),
    }

    arrays = {
        "targets": targets,
        "predictions": predictions,
        "topk_predictions": topk_predictions,
        "confidence": confidence,
        "true_action_probability": true_action_probability,
        "pass_probability": pass_probability,
    }
    return metrics, arrays


def save_predictions_csv(path: Path, arrays: dict[str, np.ndarray]) -> None:
    targets = arrays["targets"]
    predictions = arrays["predictions"]
    topk_predictions = arrays["topk_predictions"]
    confidence = arrays["confidence"]
    true_probability = arrays["true_action_probability"]
    pass_probability = arrays["pass_probability"]

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "sample_index",
                "true_action_index",
                "true_action",
                "predicted_action_index",
                "predicted_action",
                "top1_correct",
                "true_action_probability",
                "top1_confidence",
                "pass_probability",
                "top5_action_indices",
                "top5_actions",
            ],
        )
        writer.writeheader()
        for index, target in enumerate(targets):
            top5 = [int(value) for value in topk_predictions[index]]
            writer.writerow(
                {
                    "sample_index": index,
                    "true_action_index": int(target),
                    "true_action": action_name(int(target)),
                    "predicted_action_index": int(predictions[index]),
                    "predicted_action": action_name(int(predictions[index])),
                    "top1_correct": bool(predictions[index] == target),
                    "true_action_probability": float(true_probability[index]),
                    "top1_confidence": float(confidence[index]),
                    "pass_probability": float(pass_probability[index]),
                    "top5_action_indices": " ".join(map(str, top5)),
                    "top5_actions": " | ".join(action_name(value) for value in top5),
                }
            )


def plot_topk_accuracy(metrics: dict, output_dir: Path) -> None:
    names = ["Top-1", "Top-3", "Top-5"]
    values = [
        metrics["top1_accuracy"] * 100,
        metrics["top3_accuracy"] * 100,
        metrics["top5_accuracy"] * 100,
    ]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(names, values)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Next-move prediction accuracy on the test set")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1, f"{value:.2f}%", ha="center")
    fig.tight_layout()
    fig.savefig(output_dir / "topk_accuracy.png", dpi=160)
    plt.close(fig)


def plot_pass_confusion(metrics: dict, output_dir: Path) -> None:
    # This is a two-group diagnostic: did the model predict PASS versus a board move?
    matrix = np.array(
        [
            [metrics["pass_true_negatives"], metrics["pass_false_positives"]],
            [metrics["pass_false_negatives"], metrics["pass_true_positives"]],
        ],
        dtype=np.int64,
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Board move", "PASS"])
    ax.set_yticks([0, 1], labels=["Board move", "PASS"])
    ax.set_xlabel("Predicted action group")
    ax.set_ylabel("True action group")
    ax.set_title("Board move vs PASS (counts)")
    for row in range(2):
        for col in range(2):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center")
    fig.colorbar(image, ax=ax, shrink=0.8, label="Number of examples")
    fig.tight_layout()
    fig.savefig(output_dir / "pass_confusion_matrix.png", dpi=160)
    plt.close(fig)


def plot_pass_probability(arrays: dict[str, np.ndarray], output_dir: Path) -> None:
    targets = arrays["targets"]
    pass_probability = arrays["pass_probability"]
    board_values = pass_probability[targets != PASS_MOVE]
    pass_values = pass_probability[targets == PASS_MOVE]

    fig, ax = plt.subplots(figsize=(7, 5))
    datasets = []
    positions = []
    labels = []
    if board_values.size:
        datasets.append(board_values)
        positions.append(1)
        labels.append(f"True board move\n(n={board_values.size})")
    if pass_values.size:
        datasets.append(pass_values)
        positions.append(2)
        labels.append(f"True PASS\n(n={pass_values.size})")

    if datasets:
        ax.boxplot(datasets, positions=positions, widths=0.5, showmeans=True)
        ax.set_xticks(positions, labels=labels)
        # Keep individual PASS observations visible when the group is very small.
        if pass_values.size <= 100:
            rng = np.random.default_rng(0)
            jitter = rng.uniform(-0.06, 0.06, size=pass_values.size)
            ax.scatter(2 + jitter, pass_values, s=24, zorder=3, label="Individual PASS examples")
            if pass_values.size:
                ax.legend(loc="best")
    else:
        ax.text(0.5, 0.5, "No examples available", ha="center", va="center", transform=ax.transAxes)

    ax.set_ylim(0, 1)
    ax.set_ylabel("Model probability assigned to PASS")
    ax.set_title("PASS probability by ground-truth action")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "pass_probability_by_target.png", dpi=160)
    plt.close(fig)


def plot_confidence(arrays: dict[str, np.ndarray], output_dir: Path) -> None:
    correct = arrays["predictions"] == arrays["targets"]
    confidence = arrays["confidence"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(0, 1, 21)
    if np.any(correct):
        ax.hist(confidence[correct], bins=bins, alpha=0.65, label=f"Correct (n={int(correct.sum())})")
    if np.any(~correct):
        ax.hist(
            confidence[~correct],
            bins=bins,
            alpha=0.65,
            label=f"Incorrect (n={int((~correct).sum())})",
        )
    ax.set_xlabel("Maximum predicted action probability")
    ax.set_ylabel("Number of examples")
    ax.set_title("Prediction confidence: correct vs incorrect")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "prediction_confidence.png", dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative.")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not args.test_data.is_file():
        raise FileNotFoundError(f"Test dataset not found: {args.test_data}")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this PyTorch environment.")

    # The checkpoint is a local artifact produced by this project, not an untrusted file.
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if "config_dict" not in checkpoint or "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint must contain 'config_dict' and 'model_state_dict'.")

    dataset = GoDataset(str(args.test_data), augment=False)
    if len(dataset) == 0:
        raise ValueError(f"Test dataset is empty: {args.test_data}")
    sample_x, _ = dataset[0]
    if tuple(sample_x.shape) != (7, BOARD_SIZE, BOARD_SIZE):
        raise ValueError(
            f"Expected each input to have shape (7, 9, 9), received {tuple(sample_x.shape)}."
        )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = GoTRMModel(checkpoint["config_dict"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test dataset: {args.test_data}")
    print(f"Test examples: {len(dataset)}")
    print("Evaluating on test set...")

    metrics, arrays = evaluate_model(model, loader, device)
    metrics["checkpoint"] = str(args.checkpoint.resolve())
    metrics["test_data"] = str(args.test_data.resolve())
    metrics["checkpoint_epoch"] = checkpoint.get("epoch")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False, allow_nan=False)
        file.write("\n")

    save_predictions_csv(args.output_dir / "predictions.csv", arrays)
    plot_topk_accuracy(metrics, args.output_dir)
    plot_pass_confusion(metrics, args.output_dir)
    plot_pass_probability(arrays, args.output_dir)
    plot_confidence(arrays, args.output_dir)

    print("\n=== TEST METRICS ===")
    print(f"Cross-entropy loss: {metrics['cross_entropy_loss']:.4f}")
    print(f"Top-1 accuracy:     {metrics['top1_accuracy'] * 100:.2f}%")
    print(f"Top-3 accuracy:     {metrics['top3_accuracy'] * 100:.2f}%")
    print(f"Top-5 accuracy:     {metrics['top5_accuracy'] * 100:.2f}%")
    print(f"Board-only Top-1:   {metrics['board_move_top1_accuracy']}")
    print(f"PASS examples:      {metrics['pass_examples']}")
    print(f"PASS precision:     {metrics['pass_precision']}")
    print(f"PASS recall:        {metrics['pass_recall']}")
    print(f"PASS F1:            {metrics['pass_f1']}")
    if metrics["warning"]:
        print(f"WARNING: {metrics['warning']}")
    print(f"\nResults saved to: {args.output_dir.resolve()}")
    print("Generated: metrics.json, predictions.csv, and four PNG plots.")


if __name__ == "__main__":
    main()