import torch
import torch.nn as nn

class GoPolicyHead(nn.Module):
    """
    Projects the TRM latent states [Batch, 81, hidden_size] 
    into 82 move logits [Batch, 82].
    """
    def __init__(self, hidden_size=256):
        super(GoPolicyHead, self).__init__()
        
        # Linear projection for the 81 board intersections
        # Projects [Batch, 81, hidden_size] to [Batch, 81, 1]
        self.board_head = nn.Linear(hidden_size, 1)
        
        # Projection for the PASS move (index 81)
        # Applied after pooling the entire board state
        self.pass_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): TRM output state [Batch, 81, hidden_size]
        Returns:
            torch.Tensor: Raw logits before softmax [Batch, 82]
        """
        # 1. Board logits: score each intersection independently
        # Shape: [Batch, 81, 1]
        board_logits = self.board_head(x)
        
        # Squeeze the last dimension to get [Batch, 81]
        board_logits = board_logits.squeeze(-1)
        
        # 2. Pass logit: Global Average Pooling over the spatial tokens (dim=1)
        # Shape: [Batch, hidden_size]
        pooled_state = x.mean(dim=1)
        
        # Project pooled state to a single scalar per batch item
        # Shape: [Batch, 1]
        pass_logit = self.pass_head(pooled_state)
        
        # 3. Concatenate board logits and pass logit
        # Final shape: [Batch, 82]
        final_logits = torch.cat([board_logits, pass_logit], dim=-1)
        
        return final_logits

if __name__ == "__main__":
    # Sanity check
    dummy_trm_output = torch.randn(32, 81, 256)
    head = GoPolicyHead(hidden_size=256)
    logits = head(dummy_trm_output)
    print(f"Output shape: {logits.shape}")
    pass

'''output:
Output shape: torch.Size([32, 82])
'''