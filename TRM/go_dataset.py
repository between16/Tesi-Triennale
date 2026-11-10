import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from pathlib import Path

DEFAULT_BOARD_SIZE = 9


def default_go_dataset_dir(
    board_size: int,
    base_dir: str | Path | None = None,
) -> Path:
    """Return the conventional generated-dataset directory for a board size."""
    if board_size == 9:
        dataset_name = "go_dataset_go9"
    elif board_size == 19:
        dataset_name = "go_dataset_professional_10k"
    else:
        dataset_name = "go_dataset"

    trm_dir = (
        Path(base_dir)
        if base_dir is not None
        else Path(__file__).resolve().parent
    )
    return trm_dir / f"{board_size}x{board_size}_go" / dataset_name


def apply_board_symmetry(x_tensor, y_target, symmetry_idx, board_size=None):
    """
    Applies one of the 8 dihedral symmetries (D4 group) to both the input 
    state tensor X [C, N, N] and target move y (0..N*N).
    
    symmetry_idx: Integer from 0 to 7.
    board_size: Board width; inferred from x_tensor when omitted.
    """
    board_size = x_tensor.shape[-1] if board_size is None else board_size
    if not 0 <= symmetry_idx < 8:
        raise ValueError(f"Symmetry index must be in the range 0..7, got {symmetry_idx}.")
    if x_tensor.shape[-2:] != (board_size, board_size):
        raise ValueError(
            f"Expected a {board_size}x{board_size} board, received {x_tensor.shape[-2:]}."
        )
    pass_move = board_size * board_size
    if not 0 <= int(y_target) <= pass_move:
        raise ValueError(f"Target move must be in the range 0..{pass_move}.")
    if symmetry_idx == 0:
        return x_tensor, y_target

    # Extract rotation count (0..3) and flip flag (0 or 1)
    rot_k = symmetry_idx % 4
    flip_flag = symmetry_idx // 4

    # 1. Transform input state tensor X [C, N, N]
    # np.rot90 rotates along spatial axes (1, 2)
    x_transformed = np.rot90(x_tensor, k=rot_k, axes=(1, 2))
    if flip_flag:
        x_transformed = np.flip(x_transformed, axis=2)

    # Make copy to fix negative strides from numpy flip/rot operations
    x_transformed = np.ascontiguousarray(x_transformed)

    # 2. Transform target move index y
    if y_target == pass_move:
        return x_transformed, y_target

    # Map flat target index y back to 2D coordinates (row, col)
    row = y_target // board_size
    col = y_target % board_size

    # Represent coordinate as a 2D binary mask to reuse spatial numpy operations
    target_mask = np.zeros((board_size, board_size), dtype=np.uint8)
    target_mask[row, col] = 1

    target_mask = np.rot90(target_mask, k=rot_k, axes=(0, 1))
    if flip_flag:
        target_mask = np.flip(target_mask, axis=1)

    # Find the new transformed coordinate
    new_row, new_col = np.argwhere(target_mask == 1)[0]
    y_transformed = int(new_row * board_size + new_col)

    return x_transformed, y_transformed


class GoDataset(Dataset):
    """
    PyTorch Dataset for loading Go state-move pairs from NPZ or NPY files.
    """
    def __init__(self, dataset_path, augment=False, board_size=None):
        """
        Args:
            dataset_path (str): NPZ file or directory containing X.npy and y.npy.
            augment (bool): If True, applies random D4 symmetry transformations.
        """
        dataset_path = os.fspath(dataset_path)
        if os.path.isdir(dataset_path):
            self.X = np.load(os.path.join(dataset_path, "X.npy"), mmap_mode="r")
            self.y = np.load(os.path.join(dataset_path, "y.npy"), mmap_mode="r")
        else:
            data = np.load(dataset_path)
            self.X = data["X"]
            self.y = data["y"]
        if self.X.ndim != 4 or self.X.shape[-1] != self.X.shape[-2]:
            raise ValueError(f"Expected square board tensors [N, C, N, N], received {self.X.shape}.")
        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError("Feature and target arrays must contain the same number of examples.")
        self.board_size = int(self.X.shape[-1] if board_size is None else board_size)
        if self.X.shape[-2:] != (self.board_size, self.board_size):
            raise ValueError(
                f"Dataset board shape {self.X.shape[-2:]} does not match "
                f"board size {self.board_size}."
            )
        pass_move = self.board_size * self.board_size
        for start in range(0, len(self.y), 1_000_000):
            y_chunk = self.y[start : start + 1_000_000]
            if np.any(y_chunk < 0) or np.any(y_chunk > pass_move):
                raise ValueError(f"Dataset labels must be in the range 0..{pass_move}.")
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x_sample = self.X[idx]
        y_sample = self.y[idx]

        if self.augment:
            symmetry_idx = np.random.randint(0, 8)
            x_sample, y_sample = apply_board_symmetry(
                x_sample, y_sample, symmetry_idx, self.board_size
            )

        # Convert numpy arrays to PyTorch tensors
        x_tensor = torch.from_numpy(np.array(x_sample, copy=True)).float()
        y_tensor = torch.tensor(y_sample, dtype=torch.long)

        return x_tensor, y_tensor


def create_go_dataloaders(
    train_npz, val_npz, test_npz=None, batch_size=256, num_workers=4, board_size=None
):
    """
    Factory function to construct PyTorch DataLoaders for train, val, and test splits.
    """
    train_dataset = GoDataset(train_npz, augment=True, board_size=board_size)
    val_dataset = GoDataset(val_npz, augment=False, board_size=board_size)

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
        test_dataset = GoDataset(test_npz, augment=False, board_size=board_size)
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
    dataset_dir = default_go_dataset_dir(DEFAULT_BOARD_SIZE)
    train_loader, val_loader, _ = create_go_dataloaders(
        dataset_dir / "train.npz",
        dataset_dir / "val.npz",
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