import argparse
import os
import shutil
from pathlib import Path

from sgfmill import sgf


def main():
    parser = argparse.ArgumentParser(description="Extract SGF games for a board size.")
    parser.add_argument("--board-size", type=int, default=9)
    parser.add_argument("--source", type=Path, default=Path("./games"))
    parser.add_argument(
        "--destination",
        type=Path,
        help="Destination directory (defaults to the matching TRM board-size folder).",
    )
    args = parser.parse_args()
    if args.board_size < 2:
        parser.error("--board-size must be at least 2.")

    trm_dir = Path(__file__).resolve().parents[2]
    destination = args.destination or (
        trm_dir
        / f"{args.board_size}x{args.board_size}_go"
        / "go_raw_dataset"
        / f"{args.board_size}x{args.board_size}_games"
    )
    destination.mkdir(parents=True, exist_ok=True)

    count_found = 0
    count_copied = 0
    print(f"Source: {args.source} ...")

    for root, _, files in os.walk(args.source):
        for file in files:
            if file.endswith((".sgf", ".SGF")):
                filepath = Path(root) / file
                count_found += 1

                try:
                    with filepath.open("rb") as source_file:
                        game = sgf.Sgf_game.from_bytes(source_file.read())

                    if game.get_size() == args.board_size:
                        count_copied += 1
                        new_name = (
                            f"game_{args.board_size}x{args.board_size}_"
                            f"{count_copied:05d}.sgf"
                        )
                        shutil.copy2(filepath, destination / new_name)
                except Exception:
                    continue

    print(f"SGF files analyzed: {count_found}")
    print(f"{args.board_size}x{args.board_size} games in '{destination}': {count_copied}")


if __name__ == "__main__":
    main()