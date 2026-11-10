import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from models.go_model import GoTRMModel
from go_dataset import create_go_dataloaders, default_go_dataset_dir


def run_micro_train(board_size=9):
    # 1. Model configuration
    board_area = board_size * board_size
    config_dict = {
        "batch_size": 32,
        "board_size": board_size,
        "seq_len": board_area,
        "in_channels": 7,
        "vocab_size": board_area + 1,
        "H_cycles": 2,
        "L_cycles": 2,
        "H_layers": 1,
        "L_layers": 1,
        "hidden_size": 128,
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
    print(f"Running micro-training test on device: {device}")

    # 2. Instantiate unified model
    model = GoTRMModel(config_dict).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # 3. Create dummy data batch or load 1 real batch
    dataset_dir = default_go_dataset_dir(board_size)
    train_path = dataset_dir / "train"
    val_path = dataset_dir / "val"
    if not train_path.is_dir() or not val_path.is_dir():
        train_path = dataset_dir / "train.npz"
        val_path = dataset_dir / "val.npz"

    if train_path.exists() and val_path.exists():
        train_loader, _, _ = create_go_dataloaders(
            train_path,
            val_path,
            batch_size=32,
            board_size=board_size,
        )
        x_batch, y_batch = next(iter(train_loader))
        print(f"Loaded 1 real batch from {train_path}")
    elif not train_path.exists() and not val_path.exists():
        print("Dataset file not found. Generating synthetic dummy batch for sanity check...")
        x_batch = torch.randn(32, 7, board_size, board_size)
        y_batch = torch.randint(0, board_area + 1, (32,))
    else:
        missing_path = train_path if not train_path.is_file() else val_path
        raise FileNotFoundError(f"Dataset split not found: {missing_path}")

    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)

    # 4. Single-batch overfitting loop
    model.train()
    print("\n--- Starting Overfitting Test (100 Steps) ---")
    
    for step in range(1, 101):
        optimizer.zero_grad()
        
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        
        loss.backward()
        optimizer.step()
        
        # Calculate Top-1 Accuracy
        preds = logits.argmax(dim=-1)
        acc = (preds == y_batch).float().mean().item() * 100.0
        
        if step % 10 == 0 or step == 1:
            print(f"Step {step:3d}/100 | Loss: {loss.item():.4f} | Top-1 Accuracy: {acc:.2f}%")

    print("\n--- Micro-training completed successfully! ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a GoTRM single-batch overfit check.")
    parser.add_argument("--board-size", type=int, default=9)
    args = parser.parse_args()
    if args.board_size < 2:
        parser.error("--board-size must be at least 2.")
    run_micro_train(board_size=args.board_size)

'''output:

Running micro-training test on device: cuda
Loaded 1 real batch from dataset/train.npz

--- Starting Overfitting Test (100 Steps) ---
Step   1/100 | Loss: 4.4000 | Top-1 Accuracy: 0.00%
Step  10/100 | Loss: 0.8941 | Top-1 Accuracy: 90.62%
Step  20/100 | Loss: 0.0049 | Top-1 Accuracy: 100.00%
Step  30/100 | Loss: 0.0005 | Top-1 Accuracy: 100.00%
Step  40/100 | Loss: 0.0002 | Top-1 Accuracy: 100.00%
Step  50/100 | Loss: 0.0001 | Top-1 Accuracy: 100.00%
Step  60/100 | Loss: 0.0001 | Top-1 Accuracy: 100.00%
Step  70/100 | Loss: 0.0001 | Top-1 Accuracy: 100.00%
Step  80/100 | Loss: 0.0001 | Top-1 Accuracy: 100.00%
Step  90/100 | Loss: 0.0001 | Top-1 Accuracy: 100.00%
Step 100/100 | Loss: 0.0001 | Top-1 Accuracy: 100.00%

--- Micro-training completed successfully! ---
'''