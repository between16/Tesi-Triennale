import os
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
        self.policy_head = GoPolicyHead(hidden_size=config_dict["hidden_size"])

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
    _, topk_preds = logits.topk(k, dim=-1, largest=True, sorted=True)
    correct = topk_preds.eq(targets.view(-1, 1).expand_as(topk_preds))
    return correct.any(dim=-1).float().mean().item() * 100.0


def evaluate(model, val_loader, criterion, device):
    """
    Evaluates the model on the validation dataset.
    """
    model.eval()
    total_loss = 0.0
    total_top1 = 0.0
    total_top3 = 0.0
    num_batches = len(val_loader)

    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch)
            loss = criterion(logits, y_batch)

            total_loss += loss.item()
            total_top1 += calculate_topk_accuracy(logits, y_batch, k=1)
            total_top3 += calculate_topk_accuracy(logits, y_batch, k=3)

    return (
        total_loss / num_batches,
        total_top1 / num_batches,
        total_top3 / num_batches
    )


def train():
    # 1. Training Hyperparameters and Configuration
    num_epochs = 20
    batch_size = 128
    learning_rate = 1e-3
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    config_dict = {
        "batch_size": batch_size,
        "seq_len": 81,
        "in_channels": 7,
        "vocab_size": 82,
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
    train_loader, val_loader, _ = create_go_dataloaders(
        train_npz="./9x9_go/go_dataset/train.npz",
        val_npz="./9x9_go/go_dataset/val.npz",
        batch_size=batch_size,
        num_workers=4
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
        running_loss = 0.0
        running_top1 = 0.0

        for step, (x_batch, y_batch) in enumerate(train_loader, start=1):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_top1 += calculate_topk_accuracy(logits, y_batch, k=1)

            if step % 50 == 0 or step == len(train_loader):
                print(
                    f"Epoch [{epoch:2d}/{num_epochs:2d}] | "
                    f"Step [{step:4d}/{len(train_loader):4d}] | "
                    f"Train Loss: {loss.item():.4f} | "
                    f"Train Top-1 Acc: {running_top1 / step:.2f}%"
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
    train()