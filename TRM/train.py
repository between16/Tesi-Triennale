import os
import argparse
import math
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.go_trm import TinyRecursiveReasoningModel_ACTV1
from models.go_policy_head import GoPolicyHead
from go_dataset import create_go_dataloaders


class GoTRMModel(nn.Module):
    """
    Unified Go Model wrapping TRM Core and GoPolicyHead.
    """
    def __init__(self, config_dict):
        super(GoTRMModel, self).__init__()
        self.trm = TinyRecursiveReasoningModel_ACTV1(config_dict)
        self.policy_head = GoPolicyHead(
            hidden_size=config_dict["hidden_size"],
            board_size=config_dict.get("board_size", 9),
        )

    def forward(self, x):
        batch_dict = {"inputs": x}
        carry = self.trm.initial_carry(batch_dict)
        carry, hidden_features = self.trm(carry, batch_dict)
        logits = self.policy_head(hidden_features)
        return logits


def calculate_topk_accuracy(logits, targets, k=1):
    """
    Computes top-k accuracy percentage.
    """
    return calculate_topk_correct_count(logits, targets, k) / targets.numel() * 100.0


def calculate_topk_correct_count(logits, targets, k=1):
    """Return the number of examples whose target is in the top-k predictions."""
    _, topk_preds = logits.topk(k, dim=-1, largest=True, sorted=True)
    correct = topk_preds.eq(targets.view(-1, 1).expand_as(topk_preds))
    return int(correct.any(dim=-1).sum().item())


def evaluate(model, val_loader, criterion, device, max_batches=None):
    """
    Evaluates the model on the validation dataset.
    """
    model.eval()
    total_loss = 0.0
    total_top1_correct = 0
    total_top3_correct = 0
    total_examples = 0

    with torch.no_grad():
        for batch_index, (x_batch, y_batch) in enumerate(val_loader, start=1):
            if max_batches is not None and batch_index > max_batches:
                break
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            batch_size = y_batch.size(0)

            total_loss += loss.item() * batch_size
            total_top1_correct += calculate_topk_correct_count(logits, y_batch, k=1)
            total_top3_correct += calculate_topk_correct_count(logits, y_batch, k=3)
            total_examples += batch_size

    if total_examples == 0:
        raise ValueError("Validation loader did not yield any examples.")

    return (
        total_loss / total_examples,
        total_top1_correct / total_examples * 100.0,
        total_top3_correct / total_examples * 100.0
    )


def _save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    config_dict,
    epoch,
    step,
    **metrics,
):
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "epoch": epoch,
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config_dict": config_dict,
            **metrics,
        },
        temporary_path,
    )
    os.replace(temporary_path, path)


def _save_inference_checkpoint(path, model, config_dict, epoch, step, **metrics):
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "epoch": epoch,
            "step": step,
            "model_state_dict": model.state_dict(),
            "config_dict": config_dict,
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
            f"Training/validation split not found for format "
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
    max_hours=None,
    max_steps=None,
    validation_batches=None,
    checkpoint_every_steps=500,
    resume=False,
):
    # 1. Training Hyperparameters and Configuration
    num_epochs = epochs
    batch_size = batch_size if batch_size is not None else (
        16 if board_size == 19 else 128
    )
    num_workers = num_workers if num_workers is not None else (
        2 if board_size == 19 else 4
    )
    learning_rate = 1e-3
    board_area = board_size * board_size
    checkpoint_dir = Path(__file__).resolve().parent / "checkpoints"
    if board_size != 9:
        checkpoint_dir /= f"{board_size}x{board_size}_go"
    os.makedirs(checkpoint_dir, exist_ok=True)

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
        "no_ACT_continue": True
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting full training pipeline on device: {device}")

    # 2. Setup DataLoaders
    dataset_dir = Path(dataset_dir) if dataset_dir else (
        Path(__file__).resolve().parent
        / f"{board_size}x{board_size}_go"
        / "go_dataset"
    )
    train_path, val_path = resolve_dataset_paths(
        dataset_dir,
        dataset_format,
    )

    train_loader, val_loader, _ = create_go_dataloaders(
        train_npz=train_path,
        val_npz=val_path,
        batch_size=batch_size,
        num_workers=num_workers,
        board_size=board_size,
    )

    # 3. Instantiate Model, Optimizer, Scheduler, Loss
    model = GoTRMModel(config_dict).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    latest_checkpoint = checkpoint_dir / "latest_model.pt"
    start_epoch = 1
    if resume:
        if not latest_checkpoint.exists():
            raise FileNotFoundError(
                f"Cannot resume: checkpoint not found at {latest_checkpoint}"
            )
        checkpoint = torch.load(latest_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        best_val_loss = checkpoint.get("best_val_loss", best_val_loss)
        start_epoch = checkpoint["epoch"]
        print(
            f"Resuming at epoch {start_epoch}; the saved partial epoch "
            "will be restarted."
        )

    start_time = time.monotonic()
    deadline = start_time + max_hours * 3600 if max_hours is not None else None
    global_step = 0
    stopped_early = False

    # 4. Main Training Loop
    for epoch in range(start_epoch, num_epochs + 1):
        model.train()
        epoch_start_time = time.monotonic()
        running_top1_correct = 0
        running_examples = 0

        for step, (x_batch, y_batch) in enumerate(train_loader, start=1):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            global_step += 1

            running_top1_correct += calculate_topk_correct_count(logits, y_batch, k=1)
            running_examples += y_batch.size(0)

            if step % 50 == 0 or step == len(train_loader):
                print(
                    f"Epoch [{epoch:2d}/{num_epochs:2d}] | "
                    f"Step [{step:4d}/{len(train_loader):4d}] | "
                    f"Train Loss: {loss.item():.4f} | "
                    f"Train Top-1 Acc: "
                    f"{running_top1_correct / running_examples * 100.0:.2f}%"
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
                f"Val Top-1 Acc: {val_top1:.2f}% | "
                f"Val Top-3 Acc: {val_top3:.2f}%"
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
                best_val_loss=best_val_loss,
            )
            elapsed = (time.monotonic() - start_time) / 3600
            epoch_elapsed = time.monotonic() - epoch_start_time
            print(
                f"Stopping after {global_step} steps and {elapsed:.2f} hours; "
                f"current-epoch throughput "
                f"{running_examples / epoch_elapsed:.1f} examples/s; "
                f"latest checkpoint saved to {latest_checkpoint}"
            )
            break

        scheduler.step()

        # 5. Validation Step
        val_loss, val_top1, val_top3 = evaluate(
            model,
            val_loader,
            criterion,
            device,
            max_batches=validation_batches,
        )
        print("-" * 75)
        print(
            f"Epoch {epoch:2d} Validation Summary -> "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Top-1 Acc: {val_top1:.2f}% | "
            f"Val Top-3 Acc: {val_top3:.2f}%"
        )
        print("-" * 75)

        # 6. Save Checkpoint if Validation Loss Improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")
            _save_inference_checkpoint(
                Path(checkpoint_path),
                model,
                config_dict,
                epoch,
                global_step,
                val_loss=val_loss,
                val_top1=val_top1,
                best_val_loss=best_val_loss,
            )
            print(f"Checkpoint saved: {checkpoint_path} (Best Val Loss: {val_loss:.4f})")

        _save_checkpoint(
            latest_checkpoint,
            model,
            optimizer,
            scheduler,
            config_dict,
            epoch + 1,
            0,
            best_val_loss=best_val_loss,
        )
        epoch_seconds = time.monotonic() - epoch_start_time
        print(
            f"Epoch {epoch} elapsed: {epoch_seconds / 60:.1f} min; "
            f"training throughput: "
            f"{running_examples / epoch_seconds:.1f} examples/s"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GoTRM for a chosen square board size.")
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
    parser.add_argument("--max-hours", type=float)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--validation-batches", type=int)
    parser.add_argument("--checkpoint-every-steps", type=int, default=500)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.board_size < 2:
        parser.error("--board-size must be at least 2.")
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
    if (
        args.epochs < 1
        or args.checkpoint_every_steps < 1
        or (args.num_workers is not None and args.num_workers < 0)
    ):
        parser.error(
            "epochs/checkpoint interval must be positive and "
            "workers nonnegative."
        )

    validation_batches = args.validation_batches
    if args.board_size == 19 and validation_batches is None:
        validation_batches = 200
    train(
        board_size=args.board_size,
        dataset_dir=args.dataset_dir,
        dataset_format=args.dataset_format,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        max_hours=args.max_hours,
        max_steps=args.max_steps,
        validation_batches=validation_batches,
        checkpoint_every_steps=args.checkpoint_every_steps,
        resume=args.resume,
    )