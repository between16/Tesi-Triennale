import torch
import torch.nn as nn

class GoEncoder(nn.Module):
    """
    Transforms a square Go board into one sequence token per intersection.
    """
    def __init__(self, in_channels=7, hidden_size=256, board_size=9):
        """
        Args:
            in_channels (int): Number of input feature maps (e.g., 7 for history).
            hidden_size (int): The embedding dimension expected by the TRM core.
        """
        super(GoEncoder, self).__init__()
        
        # 1. Convolutional Stem: 3x3 filter with padding=1 preserves the board
        # while extracting local tactical relationships (liberties, adjacencies).
        self.conv_stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_size, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU()
        )
        self.board_size = board_size
        
        self.pos_embedding = nn.Parameter(
            torch.randn(1, board_size * board_size, hidden_size)
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input board states [Batch, Channels, N, N].
            
        Returns:
            torch.Tensor: Sequence tokens [Batch, N*N, hidden_size].
        """
        if x.ndim != 4 or x.shape[-2:] != (self.board_size, self.board_size):
            raise ValueError(
                f"Expected input shape [batch, channels, {self.board_size}, "
                f"{self.board_size}], received {tuple(x.shape)}."
            )
        # Step 1: Local feature extraction
        # Output shape: [Batch, hidden_size, N, N]
        x = self.conv_stem(x)
        
        batch_size = x.size(0)
        
        # Step 2: Flatten the spatial dimensions.
        x = x.view(batch_size, x.size(1), -1)
        
        # Step 3: Transpose to match Transformer sequence format.
        x = x.transpose(1, 2)
        
        # Step 4: Add positional awareness to the tokens.
        x = x + self.pos_embedding
        
        return x

if __name__ == "__main__":
    board_size = 9
    dummy_input = torch.randn(32, 7, board_size, board_size)
    encoder = GoEncoder(in_channels=7, hidden_size=256, board_size=board_size)
    output = encoder(dummy_input)
    print(f"Input shape: {dummy_input.shape}")  
    print(f"Output shape: {output.shape}")
    pass

'''output:
Input shape: torch.Size([32, 7, 9, 9])
Output shape: torch.Size([32, 9*9, 256])
'''