import argparse
import hashlib
from collections import Counter
from pathlib import Path

from sgfmill import sgf


def extract_games(sources, destination, board_size):
    """Copy unique, readable SGF games of the requested size into destination."""
    sources = [Path(source) for source in sources]
    destination = Path(destination)
    missing_sources = [source for source in sources if not source.exists()]
    if missing_sources:
        raise FileNotFoundError(
            "Source path(s) do not exist: "
            + ", ".join(str(source) for source in missing_sources)
        )

    destination.mkdir(parents=True, exist_ok=True)
    destination_resolved = destination.resolve()
    files_found = 0
    files_unreadable = 0
    files_wrong_size = 0
    files_duplicate = 0
    files_copied = 0
    board_sizes = Counter()
    reported_errors = 0
    seen_games = set()

    for source in sources:
        candidates = [source] if source.is_file() else source.rglob("*")
        for filepath in candidates:
            if not filepath.is_file() or filepath.suffix.lower() != ".sgf":
                continue
            if destination_resolved in filepath.resolve().parents:
                continue

            files_found += 1
            try:
                content = filepath.read_bytes()
                game = sgf.Sgf_game.from_bytes(content)
            except Exception as exc:
                files_unreadable += 1
                if reported_errors < 10:
                    print(f"[WARNING] Could not read {filepath}: {exc}")
                    reported_errors += 1
                continue

            game_size = game.get_size()
            board_sizes[game_size] += 1
            if game_size != board_size:
                files_wrong_size += 1
                continue

            digest = hashlib.sha256(content).digest()
            if digest in seen_games:
                files_duplicate += 1
                continue
            seen_games.add(digest)

            files_copied += 1
            output_path = destination / (
                f"game_{board_size}x{board_size}_{files_copied:06d}.sgf"
            )
            output_path.write_bytes(content)

    print(f"SGF files analyzed: {files_found}")
    print(f"Games by board size: {dict(sorted(board_sizes.items()))}")
    print(f"Unreadable SGF files: {files_unreadable}")
    if files_unreadable > reported_errors:
        print(
            f"[WARNING] Details shown for {reported_errors} of "
            f"{files_unreadable} unreadable SGF files."
        )
    print(f"Duplicate {board_size}x{board_size} games skipped: {files_duplicate}")
    print(f"{board_size}x{board_size} games copied to '{destination}': {files_copied}")
    return {
        "analyzed": files_found,
        "unreadable": files_unreadable,
        "wrong_size": files_wrong_size,
        "duplicates": files_duplicate,
        "copied": files_copied,
        "board_sizes": board_sizes,
    }


def main():
    parser = argparse.ArgumentParser(description="Extract SGF games for a board size.")
    parser.add_argument("--board-size", type=int, default=9)
    parser.add_argument(
        "--source",
        type=Path,
        nargs="+",
        default=[Path("./games")],
        help="One or more files/directories to scan recursively (default: ./games).",
    )
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
    extract_games(args.source, destination, args.board_size)


if __name__ == "__main__":
    main()