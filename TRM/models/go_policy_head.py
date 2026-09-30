import torch
import torch.nn as nn


class GoPolicyHead(nn.Module):
    """
    Policy Head for 9x9 Go. Transforms latent sequence features [Batch, 81, hidden_size]
    into 82 output move logits (81 board intersections + 1 PASS move).
    """
    def __init__(self, hidden_size=256):
        """
        Args:
            hidden_size (int): Dimension of the latent representations from the TRM.
        """
        super(GoPolicyHead, self).__init__()
        self.hidden_size = hidden_size

        # 1. Per-token MLP for spatial board positions (0..80)
        self.board_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )

        # 2. Global MLP for the PASS decision (81)
        self.pass_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Latent tokens from TRM core of shape [Batch, 81, hidden_size]
            
        Returns:
            torch.Tensor: Unnormalized move logits of shape [Batch, 82]
        """
        # --- Board Moves (81 logits) ---
        # Project each spatial token from hidden_size -> 1
        # Output shape: [Batch, 81, 1]
        board_logits = self.board_mlp(x)
        # Squeeze to shape: [Batch, 81]
        board_logits = board_logits.squeeze(-1)

        # --- PASS Move (1 logit) ---
        # Aggregate global board state across all 81 spatial locations
        # Output shape: [Batch, hidden_size]
        global_features = x.mean(dim=1)
        # Output shape: [Batch, 1]
        pass_logit = self.pass_mlp(global_features)

        # --- Concatenation (82 logits total) ---
        # Final shape: [Batch, 82]
        logits = torch.cat([board_logits, pass_logit], dim=-1)

        return logits


if __name__ == "__main__":
    # Sanity check to verify output dimensions
    dummy_hidden = torch.randn(32, 81, 256)
    head = GoPolicyHead(hidden_size=256)
    logits = head(dummy_hidden)
    print(f"Input shape:  {dummy_hidden.shape}")  
    print(f"Output shape: {logits.shape}")        
    pass

'''output:
Input shape:  torch.Size([32, 81, 256])
Output shape: torch.Size([32, 82])
'''