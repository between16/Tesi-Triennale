import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os

BOARD_SIZE = 9
PASS_MOVE = 81


def apply_board_symmetry(x_tensor, y_target, symmetry_idx):
    """
    Applies one of the 8 dihedral symmetries (D4 group) to both the input 
    state tensor X [C, 9, 9] and the target move y (0..81).
    
    symmetry_idx: Integer from 0 to 7.
    """
    if symmetry_idx == 0:
        return x_tensor, y_target

    # Extract rotation count (0..3) and flip flag (0 or 1)
    rot_k = symmetry_idx % 4
    flip_flag = symmetry_idx // 4

    # 1. Transform input state tensor X [C, 9, 9]
    # np.rot90 rotates along spatial axes (1, 2)
    x_transformed = np.rot90(x_tensor, k=rot_k, axes=(1, 2))
    if flip_flag:
        x_transformed = np.flip(x_transformed, axis=2)

    # Make copy to fix negative strides from numpy flip/rot operations
    x_transformed = np.ascontiguousarray(x_transformed)

    # 2. Transform target move index y
    if y_target == PASS_MOVE:
        return x_transformed, y_target

    # Map flat target index y back to 2D coordinates (row, col)
    row = y_target // BOARD_SIZE
    col = y_target % BOARD_SIZE

    # Represent coordinate as a 2D binary mask to reuse spatial numpy operations
    target_mask = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.uint8)
    target_mask[row, col] = 1

    target_mask = np.rot90(target_mask, k=rot_k, axes=(0, 1))
    if flip_flag:
        target_mask = np.flip(target_mask, axis=1)

    # Find the new transformed coordinate
    new_row, new_col = np.argwhere(target_mask == 1)[0]
    y_transformed = int(new_row * BOARD_SIZE + new_col)

    return x_transformed, y_transformed


class GoDataset(Dataset):
    """
    PyTorch Dataset for loading Go 9x9 state-move pairs from .npz files.
    """
    def __init__(self, npz_path, augment=False):
        """
        Args:
            npz_path (str): Path to the .npz file containing 'X' and 'y'.
            augment (bool): If True, applies random D4 symmetry transformations.
        """
        data = np.load(npz_path)
        self.X = data['X']  # Shape: [N, C, 9, 9]
        self.y = data['y']  # Shape: [N]
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x_sample = self.X[idx]
        y_sample = self.y[idx]

        if self.augment:
            symmetry_idx = np.random.randint(0, 8)
            x_sample, y_sample = apply_board_symmetry(x_sample, y_sample, symmetry_idx)

        # Convert numpy arrays to PyTorch tensors
        x_tensor = torch.from_numpy(x_sample).float()
        y_tensor = torch.tensor(y_sample, dtype=torch.long)

        return x_tensor, y_tensor


def create_go_dataloaders(train_npz, val_npz, test_npz=None, batch_size=256, num_workers=4):
    """
    Factory function to construct PyTorch DataLoaders for train, val, and test splits.
    """
    train_dataset = GoDataset(train_npz, augment=True)
    val_dataset = GoDataset(val_npz, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    test_loader = None
    if test_npz is not None:
        test_dataset = GoDataset(test_npz, augment=False)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False
        )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Example validation of dataset and dataloader output shapes
    dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "9x9_go", "go_dataset")
    train_loader, val_loader, _ = create_go_dataloaders(
        os.path.join(dataset_dir, "train.npz"),
        os.path.join(dataset_dir, "val.npz"),
        batch_size=32,
    )
    x_batch, y_batch = next(iter(train_loader))
    print(f"Batch X shape: {x_batch.shape}") 
    print(f"Batch y shape: {y_batch.shape}") 
    pass

'''output:
Batch X shape: torch.Size([32, 7, 9, 9])
Batch y shape: torch.Size([32])
'''