
import os
import glob
import re
from collections import Counter

import numpy as np
from sgfmill import sgf, sgf_moves


BOARD_SIZE = 9
NUM_CHANNELS = 7
PASS_MOVE = 81
TERMINAL_PASSES_REQUIRED = 2


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
    tensor = np.zeros(
        (NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE),
        dtype=np.float32,
    )

    for history_idx, board in enumerate(board_history):
        if board is None:
            continue

        player_channel = history_idx * 2
        opponent_channel = history_idx * 2 + 1

        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                stone = board.get(row, col)

                if stone == current_color:
                    tensor[player_channel, row, col] = 1.0
                elif stone is not None:
                    tensor[opponent_channel, row, col] = 1.0

    if current_color == "b":
        tensor[6, :, :] = 1.0

    return tensor


def classify_game_result(game):
    """
    Classifies the SGF result conservatively.

    Only an explicitly recorded scored result is used to infer
    that missing terminal passes may be reconstructed.
    """
    try:
        raw_result = game.get_root().get("RE")
    except KeyError:
        return None, "unknown"

    result = str(raw_result).strip()
    normalized = result.upper().replace(" ", "")

    # Scored games, including draws.
    if (
        normalized in {"0", "JIGO", "DRAW"}
        or re.fullmatch(r"[BW]\+\d+(?:\.\d+)?", normalized)
    ):
        return result, "scored"

    # Resignation: do not invent pass moves.
    if re.fullmatch(r"[BW]\+(R|RESIGN|RESIGNATION)", normalized):
        return result, "resignation"

    # Time loss or forfeiture: do not invent pass moves.
    if re.fullmatch(r"[BW]\+(T|TIME)", normalized):
        return result, "time"

    if re.fullmatch(r"[BW]\+(F|FORFEIT)", normalized):
        return result, "forfeit"

    if normalized in {"VOID", "?"}:
        return result, "unknown_or_suspended"

    return result, "other"


def opposite_color(color):
    return "w" if color == "b" else "b"


def record_move_state(
    board_history,
    board,
    color,
    target,
    x_list,
    y_list,
):
    """
    Stores the board state immediately before the given move.
    The board itself is not modified by this function.
    """
    board_history[2] = board_history[1]
    board_history[1] = board_history[0]
    board_history[0] = board.copy()

    x_list.append(extract_features(board_history, color))
    y_list.append(target)


def count_trailing_passes(processed_moves):
    """Counts consecutive PASS moves at the end of a game record."""
    count = 0

    for _, move in reversed(processed_moves):
        if move is None:
            count += 1
        else:
            break

    return count


def process_sgf_file(file_path):
    """
    Returns:
        x_list: feature tensors for recorded and reconstructed moves
        y_list: target moves in [0, 81]
        metadata: statistics and processing status
    """
    metadata = {
        "status": "ok",
        "result": None,
        "result_category": "unknown",
        "existing_passes": 0,
        "added_terminal_passes": 0,
    }

    # Read the SGF and obtain the actual initial board, including
    # any setup stones in the root node.
    try:
        with open(file_path, "rb") as file:
            game = sgf.Sgf_game.from_bytes(file.read())

        if game.get_size() != BOARD_SIZE:
            metadata["status"] = "wrong_board_size"
            return [], [], metadata

        board, moves = sgf_moves.get_setup_and_moves(game)

        result, result_category = classify_game_result(game)
        metadata["result"] = result
        metadata["result_category"] = result_category

    except Exception as exc:
        metadata["status"] = "unreadable"
        print(f"[WARNING] Could not process {file_path}: {exc}")
        return [], [], metadata

    x_list = []
    y_list = []

    # Board history ordered as [T, T-1, T-2].
    board_history = [None, None, None]
    processed_moves = []
    invalid_move = False

    for color, move in moves:
        if move is None:
            target = PASS_MOVE
        else:
            row, col = move
            target = row * BOARD_SIZE + col

        # Record the state before the move.
        record_move_state(
            board_history,
            board,
            color,
            target,
            x_list,
            y_list,
        )

        # A pass does not alter the board.
        if move is not None:
            row, col = move

            try:
                board.play(row, col, color)
            except ValueError as exc:
                # Remove the example associated with the illegal move.
                x_list.pop()
                y_list.pop()

                invalid_move = True
                metadata["status"] = "invalid_move"
                print(
                    f"[WARNING] Illegal move in {file_path}: "
                    f"{color} at {(row, col)} ({exc})"
                )
                break

        processed_moves.append((color, move))

    metadata["existing_passes"] = sum(
        move is None for _, move in processed_moves
    )

    # Reconstruct only missing terminal passes in a fully processed,
    # explicitly scored game.
    can_reconstruct = (
        not invalid_move
        and len(processed_moves) > 0
        and metadata["result_category"] == "scored"
    )

    if can_reconstruct:
        trailing_passes = count_trailing_passes(processed_moves)

        last_color = processed_moves[-1][0]
        next_color = opposite_color(last_color)

        while trailing_passes < TERMINAL_PASSES_REQUIRED:
            record_move_state(
                board_history=board_history,
                board=board,
                color=next_color,
                target=PASS_MOVE,
                x_list=x_list,
                y_list=y_list,
            )

            metadata["added_terminal_passes"] += 1
            trailing_passes += 1
            last_color = next_color
            next_color = opposite_color(last_color)

    return x_list, y_list, metadata


def build_dataset(sgf_files, output_npz_path):
    """Processes SGF games and saves their examples to an NPZ file."""
    all_x = []
    all_y = []

    game_statuses = Counter()
    result_categories = Counter()
    total_existing_passes = 0
    total_added_passes = 0

    print(
        f"\nProcessing {len(sgf_files)} SGF files "
        f"for {output_npz_path}"
    )

    for file_path in sgf_files:
        x_game, y_game, metadata = process_sgf_file(file_path)

        all_x.extend(x_game)
        all_y.extend(y_game)

        game_statuses[metadata["status"]] += 1
        result_categories[metadata["result_category"]] += 1

        total_existing_passes += metadata["existing_passes"]
        total_added_passes += metadata["added_terminal_passes"]

    x_array = np.asarray(
        all_x,
        dtype=np.float32,
    ).reshape(-1, NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE)

    y_array = np.asarray(all_y, dtype=np.int64)

    output_dir = os.path.dirname(output_npz_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    np.savez_compressed(output_npz_path, X=x_array, y=y_array)

    total_passes = int(np.count_nonzero(y_array == PASS_MOVE))

    print(f"Saved dataset to {output_npz_path}")
    print(f"X shape: {x_array.shape}")
    print(f"y shape: {y_array.shape}")
    print(f"Games by processing status: {dict(game_statuses)}")
    print(f"Games by result category: {dict(result_categories)}")
    print(f"Existing PASS moves processed: {total_existing_passes}")
    print(f"Terminal PASS moves added: {total_added_passes}")
    print(f"Total PASS examples in split: {total_passes}")

    return {
        "examples": len(y_array),
        "pass_examples": total_passes,
        "added_passes": total_added_passes,
    }


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    go_dir = os.path.join(base_dir, "9x9_go")
    sgf_dir = os.path.join(
        go_dir,
        "go_raw_dataset",
        "9x9_games",
    )

    sgf_files = sorted(
        glob.glob(
            os.path.join(sgf_dir, "**", "*.sgf"),
            recursive=True,
        )
    )

    if len(sgf_files) < 3:
        raise ValueError(
            "At least 3 SGF files are required to create "
            f"train, val, and test splits; found {len(sgf_files)}"
        )

    # Split by game, not by individual moves, to avoid placing
    # moves from the same SGF game into different dataset splits.
    rng = np.random.default_rng(seed=42)
    shuffled_files = [
        sgf_files[index]
        for index in rng.permutation(len(sgf_files))
    ]

    train_end = min(
        max(int(len(shuffled_files) * 0.8), 1),
        len(shuffled_files) - 2,
    )
    val_end = min(
        max(int(len(shuffled_files) * 0.9), train_end + 1),
        len(shuffled_files) - 1,
    )

    splits = {
        "train": shuffled_files[:train_end],
        "val": shuffled_files[train_end:val_end],
        "test": shuffled_files[val_end:],
    }

    dataset_dir = os.path.join(go_dir, "go_dataset")
    split_summary = {}

    for split_name, split_files in splits.items():
        output_path = os.path.join(
            dataset_dir,
            f"{split_name}.npz",
        )
        split_summary[split_name] = build_dataset(
            split_files,
            output_path,
        )

    print("\n=== DATASET SUMMARY ===")

    for split_name, summary in split_summary.items():
        print(
            f"{split_name}: "
            f"{summary['examples']} examples, "
            f"{summary['pass_examples']} PASS examples, "
            f"{summary['added_passes']} added PASS moves"
        )

'''
python3 go_preprocessing.py 

Processing 452 SGF files - train.npz
X shape: (21102, 7, 9, 9)
y shape: (21102,)
Games by processing status: {'ok': 452}
Games by result category: {'time': 90, 'scored': 178, 'resignation': 184}
Existing PASS moves processed: 7
Terminal PASS moves added: 352
Total PASS examples in split: 359

Processing 56 SGF files - val.npz
X shape: (2505, 7, 9, 9)
y shape: (2505,)
Games by processing status: {'ok': 56}
Games by result category: {'resignation': 29, 'scored': 23, 'time': 4}
Existing PASS moves processed: 0
Terminal PASS moves added: 46
Total PASS examples in split: 46

Processing 57 SGF files - test.npz
X shape: (2798, 7, 9, 9)
y shape: (2798,)
Games by processing status: {'ok': 57}
Games by result category: {'scored': 24, 'time': 8, 'resignation': 25}
Existing PASS moves processed: 1
Terminal PASS moves added: 48
Total PASS examples in split: 49

=== DATASET SUMMARY ===
train: 21102 examples, 359 PASS examples, 352 added PASS moves
val: 2505 examples, 46 PASS examples, 46 added PASS moves
test: 2798 examples, 49 PASS examples, 48 added PASS moves
'''