
import argparse
import os
import re
import shutil
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from sgfmill import sgf, sgf_moves

from go_dataset import default_go_dataset_dir


NUM_CHANNELS = 7
DEFAULT_BOARD_SIZE = 9
TERMINAL_PASSES_REQUIRED = 2


def default_sgf_dir(
    board_size: int,
    base_dir: str | Path | None = None,
) -> Path:
    """Return the conventional filtered-SGF directory for a board size."""
    trm_dir = (
        Path(base_dir)
        if base_dir is not None
        else Path(__file__).resolve().parent
    )
    go_dir = trm_dir / f"{board_size}x{board_size}_go" / "go_raw_dataset"
    if board_size == 9:
        return go_dir / "go9_filtered"
    if board_size == 19:
        return go_dir / "19x19_professional_games"
    return go_dir / f"{board_size}x{board_size}_games"


def extract_features(board_history, current_color, board_size=DEFAULT_BOARD_SIZE):
    """
    Constructs a [7, N, N] feature tensor from board state history.

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
        (NUM_CHANNELS, board_size, board_size),
        dtype=np.float32,
    )

    for history_idx, board in enumerate(board_history):
        if board is None:
            continue

        player_channel = history_idx * 2
        opponent_channel = history_idx * 2 + 1
        board_array = (
            board
            if isinstance(board, np.ndarray)
            else np.asarray(board.board, dtype="<U4")
        )
        tensor[player_channel] = board_array == current_color
        tensor[opponent_channel] = board_array == opposite_color(current_color)

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
    board_size=DEFAULT_BOARD_SIZE,
):
    """
    Stores the board state immediately before the given move.
    The board itself is not modified by this function.
    """
    board_history[2] = board_history[1]
    board_history[1] = board_history[0]
    board_history[0] = np.asarray(board.board, dtype="<U4")

    x_list.append(extract_features(board_history, color, board_size))
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


def process_sgf_file(file_path, board_size=DEFAULT_BOARD_SIZE, emit_warnings=True):
    """
    Returns:
        x_list: feature tensors for recorded and reconstructed moves
        y_list: target moves in [0, board_size^2]
        metadata: statistics and processing status
    """
    metadata = {
        "status": "ok",
        "result": None,
        "result_category": "unknown",
        "existing_passes": 0,
        "added_terminal_passes": 0,
        "error": None,
    }

    # Read the SGF and obtain the actual initial board, including
    # any setup stones in the root node.
    try:
        with open(file_path, "rb") as file:
            game = sgf.Sgf_game.from_bytes(file.read())

        if game.get_size() != board_size:
            metadata["status"] = "wrong_board_size"
            return [], [], metadata

        board, moves = sgf_moves.get_setup_and_moves(game)

        result, result_category = classify_game_result(game)
        metadata["result"] = result
        metadata["result_category"] = result_category

    except Exception as exc:
        metadata["status"] = "unreadable"
        metadata["error"] = str(exc)
        if emit_warnings:
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
            target = board_size * board_size
        else:
            row, col = move
            target = row * board_size + col

        # Record the state before the move.
        record_move_state(
            board_history,
            board,
            color,
            target,
            x_list,
            y_list,
            board_size,
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
                metadata["error"] = (
                    f"Illegal move by {color} at {(row, col)}: {exc}"
                )
                if emit_warnings:
                    print(f"[WARNING] {file_path}: {metadata['error']}")
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
                target=board_size * board_size,
                x_list=x_list,
                y_list=y_list,
                board_size=board_size,
            )

            metadata["added_terminal_passes"] += 1
            trailing_passes += 1
            last_color = next_color
            next_color = opposite_color(last_color)

    return x_list, y_list, metadata


def count_sgf_examples(file_path, board_size=DEFAULT_BOARD_SIZE):
    """Count examples without allocating feature tensors."""
    try:
        with open(file_path, "rb") as file:
            game = sgf.Sgf_game.from_bytes(file.read())

        if game.get_size() != board_size:
            return 0

        board, moves = sgf_moves.get_setup_and_moves(game)
        _, result_category = classify_game_result(game)
    except Exception:
        return 0

    example_count = 0
    processed_moves = []
    invalid_move = False
    for color, move in moves:
        if move is not None:
            row, col = move
            try:
                board.play(row, col, color)
            except ValueError:
                invalid_move = True
                break

        example_count += 1
        processed_moves.append((color, move))

    if (
        not invalid_move
        and processed_moves
        and result_category == "scored"
    ):
        example_count += max(
            0,
            TERMINAL_PASSES_REQUIRED - count_trailing_passes(processed_moves),
        )

    return example_count


def _write_npy_header(file_obj, dtype, shape):
    np.lib.format.write_array_header_1_0(
        file_obj,
        {
            "descr": np.lib.format.dtype_to_descr(np.dtype(dtype)),
            "fortran_order": False,
            "shape": shape,
        },
    )


def _flush_examples(x_file, y_file, x_buffer, y_buffer):
    if not x_buffer:
        return
    x_chunk = np.stack(x_buffer).astype(np.float32, copy=False)
    x_file.write(memoryview(x_chunk).cast("B"))
    y_file.write(np.asarray(y_buffer, dtype=np.int64).tobytes())
    x_buffer.clear()
    y_buffer.clear()


def build_dataset(
    sgf_files,
    output_npz_path,
    board_size=DEFAULT_BOARD_SIZE,
    expected_examples=None,
):
    """Process games into a compressed NPZ without holding a split in memory."""
    if expected_examples is None:
        expected_examples = sum(
            count_sgf_examples(file_path, board_size) for file_path in sgf_files
        )

    game_statuses = Counter()
    result_categories = Counter()
    total_existing_passes = 0
    total_added_passes = 0
    error_examples = []
    examples_written = 0
    x_buffer = []
    y_buffer = []
    output_path = Path(output_npz_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"\nProcessing {len(sgf_files)} SGF files "
        f"for {output_path} ({expected_examples} examples)"
    )

    with tempfile.TemporaryDirectory(
        dir=output_path.parent,
        prefix=f".{output_path.stem}-",
    ) as temporary_dir:
        temporary_output = Path(temporary_dir) / output_path.name
        with tempfile.TemporaryFile() as y_file:
            with zipfile.ZipFile(
                temporary_output,
                mode="w",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=6,
            ) as archive:
                with archive.open(
                    "X.npy",
                    mode="w",
                    force_zip64=True,
                ) as x_file:
                    _write_npy_header(
                        x_file,
                        np.float32,
                        (
                            expected_examples,
                            NUM_CHANNELS,
                            board_size,
                            board_size,
                        ),
                    )

                    for index, file_path in enumerate(sgf_files, start=1):
                        x_game, y_game, metadata = process_sgf_file(
                            file_path,
                            board_size,
                            emit_warnings=False,
                        )

                        game_statuses[metadata["status"]] += 1
                        result_categories[metadata["result_category"]] += 1
                        total_existing_passes += metadata["existing_passes"]
                        total_added_passes += metadata["added_terminal_passes"]
                        if metadata["error"] and len(error_examples) < 10:
                            error_examples.append(
                                f"{file_path}: {metadata['error']}"
                            )

                        x_buffer.extend(x_game)
                        y_buffer.extend(y_game)
                        examples_written += len(y_game)
                        if len(x_buffer) >= 256:
                            _flush_examples(x_file, y_file, x_buffer, y_buffer)
                        if index % 10000 == 0:
                            print(f"Processed {index}/{len(sgf_files)} SGF files")

                    _flush_examples(x_file, y_file, x_buffer, y_buffer)

                if examples_written != expected_examples:
                    raise RuntimeError(
                        "SGF example count changed between counting and "
                        f"preprocessing: expected {expected_examples}, "
                        f"wrote {examples_written}."
                    )

                y_file.flush()
                y_file.seek(0)
                with archive.open(
                    "y.npy",
                    mode="w",
                    force_zip64=True,
                ) as y_output:
                    _write_npy_header(y_output, np.int64, (examples_written,))
                    while chunk := y_file.read(1024 * 1024):
                        y_output.write(chunk)

        os.replace(temporary_output, output_path)

    total_passes = (
        total_existing_passes
        + total_added_passes
    )
    print(f"Saved dataset to {output_path}")
    print(
        f"X shape: "
        f"({examples_written}, {NUM_CHANNELS}, {board_size}, {board_size})"
    )
    print(f"y shape: ({examples_written},)")
    print(f"Games by processing status: {dict(game_statuses)}")
    print(f"Games by result category: {dict(result_categories)}")
    print(f"Existing PASS moves processed: {total_existing_passes}")
    print(f"Terminal PASS moves added: {total_added_passes}")
    print(f"Total PASS examples in split: {total_passes}")
    if error_examples:
        print(
            f"[WARNING] Errors encountered in "
            f"{sum(game_statuses[key] for key in ('unreadable', 'invalid_move'))} "
            "games; first affected files:"
        )
        for error in error_examples:
            print(f"  {error}")

    return {
        "examples": examples_written,
        "pass_examples": total_passes,
        "added_passes": total_added_passes,
    }


def build_memmap_dataset(
    sgf_files,
    output_dir,
    board_size=DEFAULT_BOARD_SIZE,
    expected_examples=None,
):
    """Write a split as memory-mappable NPY arrays without retaining it in RAM."""
    if expected_examples is None:
        expected_examples = sum(
            count_sgf_examples(file_path, board_size) for file_path in sgf_files
        )

    output_path = Path(output_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    game_statuses = Counter()
    result_categories = Counter()
    total_existing_passes = 0
    total_added_passes = 0
    error_examples = []
    examples_written = 0

    print(
        f"\nProcessing {len(sgf_files)} SGF files "
        f"for {output_path} ({expected_examples} examples)"
    )

    backup_root = Path(
        tempfile.mkdtemp(
            dir=output_path.parent,
            prefix=f".{output_path.name}-backup-",
        )
    )
    temporary_dir = Path(
        tempfile.mkdtemp(dir=output_path.parent, prefix=f".{output_path.name}-")
    )
    try:
        x_array = np.lib.format.open_memmap(
            temporary_dir / "X.npy",
            mode="w+",
            dtype=np.float32,
            shape=(
                expected_examples,
                NUM_CHANNELS,
                board_size,
                board_size,
            ),
        )
        y_array = np.lib.format.open_memmap(
            temporary_dir / "y.npy",
            mode="w+",
            dtype=np.int64,
            shape=(expected_examples,),
        )

        for index, file_path in enumerate(sgf_files, start=1):
            x_game, y_game, metadata = process_sgf_file(
                file_path,
                board_size,
                emit_warnings=False,
            )
            game_statuses[metadata["status"]] += 1
            result_categories[metadata["result_category"]] += 1
            total_existing_passes += metadata["existing_passes"]
            total_added_passes += metadata["added_terminal_passes"]
            if metadata["error"] and len(error_examples) < 10:
                error_examples.append(f"{file_path}: {metadata['error']}")

            next_index = examples_written + len(y_game)
            if next_index > expected_examples:
                raise RuntimeError(
                    "SGF example count changed between counting and "
                    f"preprocessing: expected {expected_examples}, "
                    f"at least {next_index}."
                )
            if y_game:
                x_array[examples_written:next_index] = np.asarray(
                    x_game,
                    dtype=np.float32,
                )
                y_array[examples_written:next_index] = y_game
            examples_written = next_index

            if index % 10000 == 0:
                print(f"Processed {index}/{len(sgf_files)} SGF files")

        if examples_written != expected_examples:
            raise RuntimeError(
                "SGF example count changed between counting and "
                f"preprocessing: expected {expected_examples}, "
                f"wrote {examples_written}."
            )

        x_array.flush()
        y_array.flush()
        del x_array, y_array

        if output_path.exists():
            backup_path = backup_root / output_path.name
            os.replace(output_path, backup_path)
        else:
            backup_path = None

        try:
            os.replace(temporary_dir, output_path)
        except Exception:
            if backup_path is not None:
                os.replace(backup_path, output_path)
            raise

        if backup_path is not None:
            shutil.rmtree(backup_path)

    finally:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
        if backup_root.exists():
            shutil.rmtree(backup_root)

    total_passes = total_existing_passes + total_added_passes
    print(f"Saved dataset to {output_path}")
    print(
        f"X shape: "
        f"({examples_written}, {NUM_CHANNELS}, {board_size}, {board_size})"
    )
    print(f"y shape: ({examples_written},)")
    print(f"Games by processing status: {dict(game_statuses)}")
    print(f"Games by result category: {dict(result_categories)}")
    print(f"Existing PASS moves processed: {total_existing_passes}")
    print(f"Terminal PASS moves added: {total_added_passes}")
    if error_examples:
        error_count = sum(
            game_statuses[key] for key in ("unreadable", "invalid_move")
        )
        print(
            f"[WARNING] Errors encountered in {error_count} games; "
            "first affected:"
        )
        for error in error_examples:
            print(f"  {error}")

    return {
        "examples": examples_written,
        "pass_examples": total_passes,
        "added_passes": total_added_passes,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Go datasets from SGF games.")
    parser.add_argument("--board-size", type=int, default=DEFAULT_BOARD_SIZE)
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="SGF source directory (defaults to the matching filtered-games folder).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Dataset output directory (defaults to the conventional board-size folder).",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        help="Deterministically sample at most this many games before splitting.",
    )
    parser.add_argument(
        "--storage-format",
        choices=("npz", "npy"),
        default="npz",
        help="Use compressed NPZ or memory-mappable NPY split directories.",
    )
    args = parser.parse_args()
    if args.board_size < 2:
        parser.error("--board-size must be at least 2.")
    if args.max_games is not None and args.max_games < 3:
        parser.error("--max-games must be at least 3.")

    base_dir = Path(__file__).resolve().parent
    sgf_dir = args.input_dir or default_sgf_dir(args.board_size, base_dir)
    sgf_files = sorted(
        path
        for path in sgf_dir.rglob("*")
        if path.is_file() and path.suffix.lower() == ".sgf"
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
    if args.max_games is not None:
        shuffled_files = shuffled_files[: args.max_games]
        print(
            f"Selected {len(shuffled_files)} of {len(sgf_files)} SGF files "
            f"using seed 42"
        )
    example_counts = {}
    for index, file_path in enumerate(shuffled_files, start=1):
        example_counts[file_path] = count_sgf_examples(
            file_path,
            args.board_size,
        )
        if index % 1000 == 0:
            print(f"Counted positions for {index}/{len(shuffled_files)} SGF files")

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

    dataset_dir = args.output_dir or default_go_dataset_dir(
        args.board_size, base_dir
    )
    split_summary = {}

    for split_name, split_files in splits.items():
        expected_examples = sum(
            example_counts[file_path] for file_path in split_files
        )
        if args.storage_format == "npy":
            split_output = dataset_dir / split_name
            build_function = build_memmap_dataset
        else:
            split_output = dataset_dir / f"{split_name}.npz"
            build_function = build_dataset
        split_summary[split_name] = build_function(
            split_files,
            split_output,
            board_size=args.board_size,
            expected_examples=expected_examples,
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
X shape: (examples, 7, N, N)
y shape: (21102,)
Games by processing status: {'ok': 452}
Games by result category: {'time': 90, 'scored': 178, 'resignation': 184}
Existing PASS moves processed: 7
Terminal PASS moves added: 352
Total PASS examples in split: 359

Processing 56 SGF files - val.npz
X shape: (examples, 7, N, N)
y shape: (2505,)
Games by processing status: {'ok': 56}
Games by result category: {'resignation': 29, 'scored': 23, 'time': 4}
Existing PASS moves processed: 0
Terminal PASS moves added: 46
Total PASS examples in split: 46

Processing 57 SGF files - test.npz
X shape: (examples, 7, N, N)
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