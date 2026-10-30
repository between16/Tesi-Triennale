import torch
import torch.nn as nn


class GoPolicyHead(nn.Module):
    """
    Transforms one latent token per intersection into N*N board logits and PASS.
    """
    def __init__(self, hidden_size=256, board_size=9):
        """
        Args:
            hidden_size (int): Dimension of the latent representations from the TRM.
        """
        super(GoPolicyHead, self).__init__()
        self.hidden_size = hidden_size
        self.board_size = board_size

        # 1. Per-token MLP for spatial board positions (0..80)
        self.board_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )

        # Global MLP for the PASS decision.
        self.pass_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Latent tokens [Batch, N*N, hidden_size].
            
        Returns:
            torch.Tensor: Unnormalized move logits [Batch, N*N+1].
        """
        if x.ndim != 3 or x.shape[1] != self.board_size * self.board_size:
            raise ValueError(
                f"Expected {self.board_size * self.board_size} board tokens, "
                f"received {tuple(x.shape)}."
            )

        # Project each spatial token from hidden_size -> 1.
        board_logits = self.board_mlp(x)
        board_logits = board_logits.squeeze(-1)

        # Aggregate the global board state for the PASS decision.
        global_features = x.mean(dim=1)
        pass_logit = self.pass_mlp(global_features)

        logits = torch.cat([board_logits, pass_logit], dim=-1)

        return logits


if __name__ == "__main__":
    board_size = 9
    dummy_hidden = torch.randn(32, board_size * board_size, 256)
    head = GoPolicyHead(hidden_size=256, board_size=board_size)
    logits = head(dummy_hidden)
    print(f"Input shape:  {dummy_hidden.shape}")  
    print(f"Output shape: {logits.shape}")        
    pass

'''output:
Input shape:  torch.Size([32, N*N, 256])
Output shape: torch.Size([32, N*N+1])
'''