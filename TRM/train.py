import os
import argparse
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


def evaluate(model, val_loader, criterion, device):
    """
    Evaluates the model on the validation dataset.
    """
    model.eval()
    total_loss = 0.0
    total_top1_correct = 0
    total_top3_correct = 0
    total_examples = 0

    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            batch_size = y_batch.size(0)

            total_loss += loss.item() * batch_size
            total_top1_correct += calculate_topk_correct_count(logits, y_batch, k=1)
            total_top3_correct += calculate_topk_correct_count(logits, y_batch, k=3)
            total_examples += batch_size

    return (
        total_loss / total_examples,
        total_top1_correct / total_examples * 100.0,
        total_top3_correct / total_examples * 100.0
    )


def train(board_size=9):
    # 1. Training Hyperparameters and Configuration
    num_epochs = 20
    batch_size = 128
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
    dataset_dir = Path(__file__).resolve().parent / f"{board_size}x{board_size}_go" / "go_dataset"
    train_loader, val_loader, _ = create_go_dataloaders(
        train_npz=dataset_dir / "train.npz",
        val_npz=dataset_dir / "val.npz",
        batch_size=batch_size,
        num_workers=4,
        board_size=board_size,
    )

    # 3. Instantiate Model, Optimizer, Scheduler, Loss
    model = GoTRMModel(config_dict).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")

    # 4. Main Training Loop
    for epoch in range(1, num_epochs + 1):
        model.train()
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

        scheduler.step()

        # 5. Validation Step
        val_loss, val_top1, val_top3 = evaluate(model, val_loader, criterion, device)
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
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config_dict": config_dict,
                    "val_loss": val_loss,
                    "val_top1": val_top1,
                },
                checkpoint_path
            )
            print(f"Checkpoint saved: {checkpoint_path} (Best Val Loss: {val_loss:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GoTRM for a chosen square board size.")
    parser.add_argument("--board-size", type=int, default=9)
    args = parser.parse_args()
    if args.board_size < 2:
        parser.error("--board-size must be at least 2.")
    train(board_size=args.board_size)