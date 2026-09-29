import torch
import torch.nn as nn

class GoEncoder(nn.Module):
    """
    Transforms the raw [Batch, 7, 9, 9] board representation into a sequence
    of 81 tokens [Batch, 81, hidden_size] ready for the TRM architecture.
    """
    def __init__(self, in_channels=7, hidden_size=256):
        """
        Args:
            in_channels (int): Number of input feature maps (e.g., 7 for history).
            hidden_size (int): The embedding dimension expected by the TRM core.
        """
        super(GoEncoder, self).__init__()
        
        # 1. Convolutional Stem: 3x3 filter with padding=1 preserves the 9x9 grid
        # while extracting local tactical relationships (liberties, adjacencies).
        self.conv_stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_size, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU()
        )
        
        # 2. Learnable Positional Embeddings
        # A matrix of 81 vectors, one for each intersection on the 9x9 board.
        # This injects global spatial coordinates into the flattened tokens.
        self.pos_embedding = nn.Parameter(torch.randn(1, 81, hidden_size))

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input board states of shape [Batch, Channels, 9, 9]
            
        Returns:
            torch.Tensor: Sequence tokens of shape [Batch, 81, hidden_size]
        """
        # Step 1: Local feature extraction
        # Output shape: [Batch, hidden_size, 9, 9]
        x = self.conv_stem(x)
        
        batch_size = x.size(0)
        
        # Step 2: Flatten the spatial dimensions (9x9 -> 81)
        # Output shape: [Batch, hidden_size, 81]
        x = x.view(batch_size, x.size(1), -1)
        
        # Step 3: Transpose to match Transformer sequence format
        # Output shape: [Batch, 81, hidden_size]
        x = x.transpose(1, 2)
        
        # Step 4: Add positional awareness to the tokens
        # Broadcasting the [1, 81, hidden_size] parameter across the batch
        x = x + self.pos_embedding
        
        return x

if __name__ == "__main__":
    # Sanity check to verify shapes
    dummy_input = torch.randn(32, 7, 9, 9)
    encoder = GoEncoder(in_channels=7, hidden_size=256)
    output = encoder(dummy_input)
    print(f"Input shape: {dummy_input.shape}")   # Expected: [32, 7, 9, 9]
    print(f"Output shape: {output.shape}")       # Expected: [32, 81, 256]
    pass

'''output:
Input shape: torch.Size([32, 7, 9, 9])
Output shape: torch.Size([32, 81, 256])
'''