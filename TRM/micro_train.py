import torch
import torch.nn as nn
import torch.optim as optim

from models.go_trm import TinyRecursiveReasoningModel_ACTV1
from models.go_policy_head import GoPolicyHead
from go_dataset import create_go_dataloaders


class GoTRMModel(nn.Module):
    """
    Complete End-to-End Go Model:
    GoEncoder (inside TRM) -> TRM Core -> GoPolicyHead
    """
    def __init__(self, config_dict):
        super(GoTRMModel, self).__init__()
        self.trm = TinyRecursiveReasoningModel_ACTV1(config_dict)
        self.policy_head = GoPolicyHead(hidden_size=config_dict["hidden_size"])

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input board states of shape [Batch, 7, 9, 9]
            
        Returns:
            torch.Tensor: Move logits of shape [Batch, 82]
        """
        batch_dict = {"inputs": x}
        carry = self.trm.initial_carry(batch_dict)
        carry, hidden_features = self.trm(carry, batch_dict)
        logits = self.policy_head(hidden_features)
        return logits


def run_micro_train():
    # 1. Model configuration
    config_dict = {
        "batch_size": 32,
        "seq_len": 81,
        "in_channels": 7,
        "vocab_size": 82,
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
    try:
        train_loader, _, _ = create_go_dataloaders("./9x9_go/go_dataset/train.npz", "./9x9_go/go_dataset/val.npz", batch_size=32)
        x_batch, y_batch = next(iter(train_loader))
        print("Loaded 1 real batch from dataset/train.npz")
    except Exception:
        print("Dataset file not found. Generating synthetic dummy batch for sanity check...")
        x_batch = torch.randn(32, 7, 9, 9)
        y_batch = torch.randint(0, 82, (32,))

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
    run_micro_train()

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