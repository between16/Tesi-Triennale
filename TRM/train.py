import os
import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from models.go_trm import TinyRecursiveReasoningModel_ACTV1
from models.go_policy_head import GoPolicyHead
from go_dataset import create_go_dataloaders


SCHEDULER_TYPE = "warmup_cosine_per_step_v1"


class GoTRMModel(nn.Module):
    """Unified Go model wrapping the TRM core and the Go policy head."""

    def __init__(self, config_dict):
        super().__init__()
        self.trm = TinyRecursiveReasoningModel_ACTV1(config_dict)
        self.policy_head = GoPolicyHead(
            hidden_size=config_dict["hidden_size"],
            board_size=config_dict.get("board_size", 9),
        )

    def forward(self, x):
        batch_dict = {"inputs": x}
        carry = self.trm.initial_carry(batch_dict)
        carry, hidden_features = self.trm(carry, batch_dict)
        return self.policy_head(hidden_features)


def calculate_topk_correct_count(logits, targets, k=1):
    """Return the number of examples whose target is in the top-k predictions."""
    _, topk_preds = logits.topk(k, dim=-1, largest=True, sorted=True)
    correct = topk_preds.eq(targets.view(-1, 1).expand_as(topk_preds))
    return int(correct.any(dim=-1).sum().item())


def evaluate(model, val_loader, criterion, device, max_batches=None):
    """Evaluate loss, top-1 accuracy, and top-3 accuracy on validation data."""
    model.eval()
    total_loss = 0.0
    total_top1_correct = 0
    total_top3_correct = 0
    total_examples = 0

    with torch.inference_mode():
        for batch_index, (x_batch, y_batch) in enumerate(val_loader, start=1):
            if max_batches is not None and batch_index > max_batches:
                break

            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            batch_size = y_batch.size(0)

            total_loss += loss.item() * batch_size
            total_top1_correct += calculate_topk_correct_count(
                logits, y_batch, k=1
            )
            total_top3_correct += calculate_topk_correct_count(
                logits, y_batch, k=3
            )
            total_examples += batch_size

    if total_examples == 0:
        raise ValueError("Validation loader did not yield any examples.")

    return (
        total_loss / total_examples,
        total_top1_correct / total_examples * 100.0,
        total_top3_correct / total_examples * 100.0,
    )


def build_warmup_cosine_scheduler(
    optimizer,
    total_steps,
    warmup_steps,
    min_learning_rate_ratio,
):
    """Create a per-optimizer-step linear-warmup + cosine-decay scheduler."""
    total_steps = max(int(total_steps), 1)
    warmup_steps = min(int(warmup_steps), max(total_steps - 1, 0))

    def lr_lambda(current_step):
        # LambdaLR evaluates this at step 0 during initialization. Starting at
        # 1 / warmup_steps avoids a zero-learning-rate first optimizer update.
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps)

        decay_steps = max(total_steps - warmup_steps, 1)
        progress = (current_step - warmup_steps) / decay_steps
        progress = min(max(progress, 0.0), 1.0)
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_learning_rate_ratio + (
            1.0 - min_learning_rate_ratio
        ) * cosine_factor

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def _save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    config_dict,
    epoch,
    step,
    global_step,
    **metrics,
):
    """Atomically save a resumable training checkpoint."""
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "epoch": epoch,
            "step": step,
            "global_step": global_step,
            "scheduler_type": SCHEDULER_TYPE,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config_dict": config_dict,
            **metrics,
        },
        temporary_path,
    )
    os.replace(temporary_path, path)


def _save_inference_checkpoint(
    path, model, config_dict, epoch, step, **metrics
):
    """Save only what is needed to load the model for inference."""
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config_dict": config_dict,
            "epoch": epoch,
            "step": step,
            **metrics,
        },
        temporary_path,
    )
    os.replace(temporary_path, path)


def resolve_dataset_paths(dataset_dir, dataset_format="auto"):
    """Resolve train/validation splits, preferring NPY directories in auto mode."""
    dataset_dir = Path(dataset_dir)
    train_npy_path = dataset_dir / "train"
    val_npy_path = dataset_dir / "val"
    train_npz_path = dataset_dir / "train.npz"
    val_npz_path = dataset_dir / "val.npz"

    if dataset_format == "npy":
        train_path, val_path = train_npy_path, val_npy_path
    elif dataset_format == "npz":
        train_path, val_path = train_npz_path, val_npz_path
    elif dataset_format == "auto":
        if train_npy_path.is_dir() and val_npy_path.is_dir():
            train_path, val_path = train_npy_path, val_npy_path
        else:
            train_path, val_path = train_npz_path, val_npz_path
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_format}")

    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(
            "Training/validation split not found for format "
            f"'{dataset_format}' under {dataset_dir}."
        )
    return train_path, val_path


def train(
    board_size=9,
    dataset_dir=None,
    dataset_format="auto",
    batch_size=None,
    num_workers=None,
    epochs=20,
    learning_rate=1e-4,
    min_learning_rate=1e-5,
    warmup_steps=2000,
    weight_decay=1e-4,
    beta1=0.9,
    beta2=0.95,
    max_hours=None,
    max_steps=None,
    validation_batches=None,
    checkpoint_every_steps=500,
    resume=False,
):
    """Train the Go TRM with supervised next-move prediction."""
    batch_size = batch_size if batch_size is not None else (
        16 if board_size == 19 else 128
    )
    num_workers = num_workers if num_workers is not None else (
        2 if board_size == 19 else 4
    )

    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")
    if min_learning_rate <= 0 or min_learning_rate > learning_rate:
        raise ValueError(
            "min_learning_rate must be positive and no greater than learning_rate."
        )
    if warmup_steps < 0:
        raise ValueError("warmup_steps cannot be negative.")
    if weight_decay < 0:
        raise ValueError("weight_decay cannot be negative.")
    if not (0.0 <= beta1 < 1.0 and 0.0 <= beta2 < 1.0):
        raise ValueError("beta1 and beta2 must be in the range [0, 1).")

    board_area = board_size * board_size
    checkpoint_dir = Path(__file__).resolve().parent / "checkpoints"
    if board_size != 9:
        checkpoint_dir /= f"{board_size}x{board_size}_go"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    config_dict = {
        "batch_size": batch_size,
        "board_size": board_size,
        "seq_len": board_area,
        "in_channels": 7,
        "vocab_size": board_area + 1,
        "H_cycles": 2,
        "L_cycles": 2,
        "H_layers": 1,
        "L_layers": 1,
        "hidden_size": 256,
        "expansion": 2.0,
        "num_heads": 4,
        "pos_encodings": "none",
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "halt_max_steps": 1,
        "halt_exploration_prob": 0.0,
        "forward_dtype": "float32",
        "mlp_t": False,
        "puzzle_emb_len": 0,
        "no_ACT_continue": True,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting training on device: {device}")

    dataset_dir = Path(dataset_dir) if dataset_dir else (
        Path(__file__).resolve().parent
        / f"{board_size}x{board_size}_go"
        / "go_dataset"
    )
    train_path, val_path = resolve_dataset_paths(dataset_dir, dataset_format)

    train_loader, val_loader, _ = create_go_dataloaders(
        train_npz=train_path,
        val_npz=val_path,
        batch_size=batch_size,
        num_workers=num_workers,
        board_size=board_size,
    )

    steps_per_epoch = len(train_loader)
    if steps_per_epoch == 0:
        raise ValueError(
            "The training split is too small for the selected batch size."
        )
    total_training_steps = steps_per_epoch * epochs
    effective_warmup_steps = min(
        warmup_steps, max(total_training_steps - 1, 0)
    )
    min_learning_rate_ratio = min_learning_rate / learning_rate

    model = GoTRMModel(config_dict).to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(beta1, beta2),
        weight_decay=weight_decay,
    )
    scheduler = build_warmup_cosine_scheduler(
        optimizer=optimizer,
        total_steps=total_training_steps,
        warmup_steps=effective_warmup_steps,
        min_learning_rate_ratio=min_learning_rate_ratio,
    )
    criterion = nn.CrossEntropyLoss()

    print(f"Board size: {board_size}x{board_size}")
    print(f"Training examples: {len(train_loader.dataset):,}")
    print(f"Validation examples: {len(val_loader.dataset):,}")
    print(f"Batch size: {batch_size}")
    print(f"Steps per epoch: {steps_per_epoch:,}")
    print(f"Planned optimizer steps: {total_training_steps:,}")
    print(f"Optimizer: AdamW (betas={beta1}, {beta2})")
    print(f"Learning rate: {learning_rate:.2e} -> {min_learning_rate:.2e}")
    print(f"Warm-up steps: {effective_warmup_steps:,}")
    print(f"Weight decay: {weight_decay:.2e}")
    if validation_batches is None:
        print("Validation: full validation split")
    else:
        print(f"Validation: at most {validation_batches:,} batches")

    best_val_loss = float("inf")
    latest_checkpoint = checkpoint_dir / "latest_model.pt"
    start_epoch = 1
    global_step = 0

    if resume:
        if not latest_checkpoint.exists():
            raise FileNotFoundError(
                f"Cannot resume: checkpoint not found at {latest_checkpoint}"
            )

        checkpoint = torch.load(latest_checkpoint, map_location=device)
        if checkpoint.get("scheduler_type") != SCHEDULER_TYPE:
            raise ValueError(
                "The latest checkpoint was created with a different scheduler "
                "or an older version of train.py. Start a fresh run without "
                "--resume, or restore a checkpoint created by this version."
            )

        saved_config = checkpoint.get("config_dict", {})
        saved_board_size = saved_config.get("board_size")
        if saved_board_size is not None and saved_board_size != board_size:
            raise ValueError(
                f"Checkpoint board size is {saved_board_size}, but the requested "
                f"board size is {board_size}."
            )

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        best_val_loss = checkpoint.get("best_val_loss", best_val_loss)
        start_epoch = checkpoint["epoch"]
        global_step = checkpoint.get("global_step", 0)
        print(
            f"Resuming at epoch {start_epoch}; global optimizer step "
            f"{global_step:,}. If the saved epoch was incomplete, it is "
            "restarted from its beginning."
        )

    if start_epoch > epochs:
        raise ValueError(
            f"Checkpoint is already at epoch {start_epoch}, but --epochs is "
            f"{epochs}. Increase --epochs to continue training."
        )
    if max_steps is not None and global_step >= max_steps:
        raise ValueError(
            f"Checkpoint already reached max_steps={max_steps} "
            f"(global_step={global_step}). Set a larger --max-steps value."
        )

    start_time = time.monotonic()
    deadline = (
        start_time + max_hours * 3600 if max_hours is not None else None
    )
    stopped_early = False

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        epoch_start_time = time.monotonic()
        running_loss = 0.0
        running_top1_correct = 0
        running_examples = 0

        for step, (x_batch, y_batch) in enumerate(train_loader, start=1):
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            lr_used = optimizer.param_groups[0]["lr"]
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            scheduler.step()
            global_step += 1

            batch_size_actual = y_batch.size(0)
            running_loss += loss.item() * batch_size_actual
            running_top1_correct += calculate_topk_correct_count(
                logits.detach(), y_batch, k=1
            )
            running_examples += batch_size_actual

            if step % 50 == 0 or step == steps_per_epoch:
                average_loss = running_loss / running_examples
                train_top1 = running_top1_correct / running_examples * 100.0
                print(
                    f"Epoch [{epoch:2d}/{epochs:2d}] | "
                    f"Step [{step:5d}/{steps_per_epoch:5d}] | "
                    f"Global step [{global_step:7d}] | "
                    f"Train Loss: {average_loss:.4f} | "
                    f"Train Top-1: {train_top1:.2f}% | "
                    f"LR: {lr_used:.2e}"
                )

            if global_step % checkpoint_every_steps == 0:
                _save_checkpoint(
                    latest_checkpoint,
                    model,
                    optimizer,
                    scheduler,
                    config_dict,
                    epoch,
                    step,
                    global_step,
                    best_val_loss=best_val_loss,
                )

            if max_steps is not None and global_step >= max_steps:
                stopped_early = True
                break
            if deadline is not None and time.monotonic() >= deadline:
                stopped_early = True
                break

        if stopped_early:
            _save_checkpoint(
                latest_checkpoint,
                model,
                optimizer,
                scheduler,
                config_dict,
                epoch,
                step,
                global_step,
                best_val_loss=best_val_loss,
            )

            val_loss, val_top1, val_top3 = evaluate(
                model,
                val_loader,
                criterion,
                device,
                max_batches=validation_batches,
            )
            print(
                f"Stopped-run validation -> Val Loss: {val_loss:.4f} | "
                f"Val Top-1: {val_top1:.2f}% | "
                f"Val Top-3: {val_top3:.2f}%"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                _save_inference_checkpoint(
                    checkpoint_dir / "best_model.pt",
                    model,
                    config_dict,
                    epoch,
                    global_step,
                    val_loss=val_loss,
                    val_top1=val_top1,
                    val_top3=val_top3,
                    best_val_loss=best_val_loss,
                )
                print(
                    "Inference checkpoint saved: "
                    f"{checkpoint_dir / 'best_model.pt'}"
                )

            _save_checkpoint(
                latest_checkpoint,
                model,
                optimizer,
                scheduler,
                config_dict,
                epoch,
                step,
                global_step,
                best_val_loss=best_val_loss,
            )
            elapsed_hours = (time.monotonic() - start_time) / 3600.0
            epoch_seconds = time.monotonic() - epoch_start_time
            print(
                f"Stopping after {global_step:,} optimizer steps and "
                f"{elapsed_hours:.2f} hours. Current-epoch throughput: "
                f"{running_examples / max(epoch_seconds, 1e-9):.1f} "
                f"examples/s. Latest checkpoint: {latest_checkpoint}"
            )
            break

        val_loss, val_top1, val_top3 = evaluate(
            model,
            val_loader,
            criterion,
            device,
            max_batches=validation_batches,
        )
        print("-" * 78)
        print(
            f"Epoch {epoch:2d} Validation -> "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Top-1: {val_top1:.2f}% | "
            f"Val Top-3: {val_top3:.2f}%"
        )
        print("-" * 78)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            _save_inference_checkpoint(
                checkpoint_dir / "best_model.pt",
                model,
                config_dict,
                epoch,
                global_step,
                val_loss=val_loss,
                val_top1=val_top1,
                val_top3=val_top3,
                best_val_loss=best_val_loss,
            )
            print(
                f"Best inference checkpoint saved: "
                f"{checkpoint_dir / 'best_model.pt'} "
                f"(Val Loss: {best_val_loss:.4f})"
            )

        _save_checkpoint(
            latest_checkpoint,
            model,
            optimizer,
            scheduler,
            config_dict,
            epoch + 1,
            0,
            global_step,
            best_val_loss=best_val_loss,
        )

        epoch_seconds = time.monotonic() - epoch_start_time
        print(
            f"Epoch {epoch} elapsed: {epoch_seconds / 60.0:.1f} min; "
            f"training throughput including validation: "
            f"{running_examples / max(epoch_seconds, 1e-9):.1f} examples/s"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train GoTRM for a chosen square board size."
    )
    parser.add_argument("--board-size", type=int, default=9)
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument(
        "--dataset-format",
        choices=("auto", "npy", "npz"),
        default="auto",
        help="Split format; auto prefers memory-mapped NPY when both are present.",
    )
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--max-hours", type=float)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument(
        "--validation-batches",
        type=int,
        help="Optional validation limit. Omit it to evaluate the full validation split.",
    )
    parser.add_argument("--checkpoint-every-steps", type=int, default=500)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.board_size < 2:
        parser.error("--board-size must be at least 2.")
    if args.epochs < 1:
        parser.error("--epochs must be positive.")
    if args.max_hours is not None and (
        not math.isfinite(args.max_hours) or args.max_hours <= 0
    ):
        parser.error("--max-hours must be a positive, finite number.")
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("--max-steps must be positive.")
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("--batch-size must be positive.")
    if args.validation_batches is not None and args.validation_batches < 1:
        parser.error("--validation-batches must be positive.")
    if args.checkpoint_every_steps < 1:
        parser.error("--checkpoint-every-steps must be positive.")
    if args.warmup_steps < 0:
        parser.error("--warmup-steps cannot be negative.")
    if args.num_workers is not None and args.num_workers < 0:
        parser.error("--num-workers cannot be negative.")

    train(
        board_size=args.board_size,
        dataset_dir=args.dataset_dir,
        dataset_format=args.dataset_format,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        beta1=args.beta1,
        beta2=args.beta2,
        max_hours=args.max_hours,
        max_steps=args.max_steps,
        validation_batches=args.validation_batches,
        checkpoint_every_steps=args.checkpoint_every_steps,
        resume=args.resume,
    )
