import os
import glob
import numpy as np
from sgfmill import sgf, boards, sgf_moves

BOARD_SIZE = 9
NUM_CHANNELS = 7
PASS_MOVE = 81


def extract_features(board_history, current_color):
    """
    Constructs a [7, 9, 9] feature tensor from board state history.
    
    Channels:
    0: Player stones at T
    1: Opponent stones at T
    2: Player stones at T-1
    3: Opponent stones at T-1
    4: Player stones at T-2
    5: Opponent stones at T-2
    6: Color channel (1.0 for Black, 0.0 for White)
    """
    tensor = np.zeros((NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    
    # Fill history channels (T, T-1, T-2)
    for history_idx, board in enumerate(board_history):
        if board is None:
            continue
            
        player_channel = history_idx * 2
        opponent_channel = history_idx * 2 + 1
        
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                stone = board.get(r, c)
                if stone == current_color:
                    tensor[player_channel, r, c] = 1.0
                elif stone is not None:
                    tensor[opponent_channel, r, c] = 1.0

    # Fill color to move channel
    if current_color == 'b':
        tensor[6, :, :] = 1.0

    return tensor


def process_sgf_file(file_path):
    """
    Parses a single SGF file and returns X and y arrays for all played moves.
    """
    with open(file_path, 'rb') as f:
        game = sgf.Sgf_game.from_bytes(f.read())

    if game.get_size() != BOARD_SIZE:
        return [], []

    board = boards.Board(BOARD_SIZE)
    _, moves = sgf_moves.get_setup_and_moves(game)

    x_list = []
    y_list = []

    # Maintain history of last 3 board states [T, T-1, T-2]
    board_history = [None, None, None]

    for color, move in moves:
        # Update board history shift
        board_history[2] = board_history[1]
        board_history[1] = board_history[0]
        board_history[0] = board.copy()

        # Extract 7-channel state before playing the move
        x_tensor = extract_features(board_history, color)
        x_list.append(x_tensor)

        # Encode target move
        if move is None:
            y_list.append(PASS_MOVE)
        else:
            row, col = move
            y_list.append(row * BOARD_SIZE + col)

        # Apply move to board
        if move is not None:
            try:
                board.play(row, col, color)
            except ValueError:
                # Handle illegal moves gracefully if SGF is corrupted
                x_list.pop()
                y_list.pop()
                break

    return x_list, y_list


def build_dataset(sgf_files, output_npz_path):
    """
    Processes the supplied SGF files and saves X and y arrays into an .npz file.
    Creates the output directory automatically if it does not exist.
    """
    all_x = []
    all_y = []

    print(f"Processing {len(sgf_files)} SGF files for {output_npz_path}")

    for file_path in sgf_files:
        x_game, y_game = process_sgf_file(file_path)
        all_x.extend(x_game)
        all_y.extend(y_game)

    x_array = np.array(all_x, dtype=np.float32)
    y_array = np.array(all_y, dtype=np.int64)

    # Create the destination directory if it doesn't exist
    output_dir = os.path.dirname(output_npz_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    np.savez_compressed(output_npz_path, X=x_array, y=y_array)
    print(f"Saved dataset to {output_npz_path}")
    print(f"Dataset shape - X: {x_array.shape}, y: {y_array.shape}")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    go_dir = os.path.join(base_dir, "9x9_go")
    sgf_dir = os.path.join(go_dir, "go_raw_dataset", "9x9_games")
    sgf_files = sorted(glob.glob(os.path.join(sgf_dir, "**", "*.sgf"), recursive=True))

    if len(sgf_files) < 3:
        raise ValueError(f"At least 3 SGF files are required to create train, val, and test splits; found {len(sgf_files)} in {sgf_dir}")

    # Shuffle games reproducibly so moves from one game stay in the same split.
    rng = np.random.default_rng(seed=42)
    shuffled_files = [sgf_files[index] for index in rng.permutation(len(sgf_files))]
    train_end = min(max(int(len(shuffled_files) * 0.8), 1), len(shuffled_files) - 2)
    val_end = min(max(int(len(shuffled_files) * 0.9), train_end + 1), len(shuffled_files) - 1)

    splits = {
        "train": shuffled_files[:train_end],
        "val": shuffled_files[train_end:val_end],
        "test": shuffled_files[val_end:],
    }

    dataset_dir = os.path.join(go_dir, "go_dataset")
    for split, split_files in splits.items():
        build_dataset(split_files, os.path.join(dataset_dir, f"{split}.npz"))


''' output:
Processing 452 SGF files for TRM/9x9_go/go_dataset/train.npz
Saved dataset to TRM/9x9_go/go_dataset/train.npz
Dataset shape - X: (20750, 7, 9, 9), y: (20750,)
Processing 56 SGF files for TRM/9x9_go/go_dataset/val.npz
Saved dataset to TRM/9x9_go/go_dataset/val.npz
Dataset shape - X: (2459, 7, 9, 9), y: (2459,)
Processing 57 SGF files for TRM/9x9_go/go_dataset/test.npz
Saved dataset to TRM/9x9_go/go_dataset/test.npz
Dataset shape - X: (2750, 7, 9, 9), y: (2750,)
'''