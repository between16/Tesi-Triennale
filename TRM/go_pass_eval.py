"""Evaluate whether the TRM ends self-play games with its own PASS decisions.

Run from the TRM directory:
    python3 go_pass_eval.py --games 1 --selection greedy

For varied exploratory games:
    python3 go_pass_eval.py --games 20 --selection sample --temperature 1.0

This script deliberately does NOT auto-play the opponent's PASS. A game ends
only when GoGameEnv receives two consecutive PASS actions predicted by the model.
The move limit is only a safety cutoff; reaching it is NOT counted as a game end.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

# Print before importing PyTorch or project modules, so a slow import is visible.
print("[PASS-EVAL] Script started; importing NumPy and PyTorch...", flush=True)

import numpy as np
import torch

print("[PASS-EVAL] NumPy and PyTorch imported; loading local Go modules...", flush=True)

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from agents import TRMAgent
from game_env import GoGameEnv

print("[PASS-EVAL] Local Go modules imported.", flush=True)

BOARD_SIZE = 9
KOMI = 7.5
PASS_MOVE = BOARD_SIZE * BOARD_SIZE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test whether the TRM naturally ends 9x9 Go self-play games with PASS."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=SCRIPT_DIR / "checkpoints" / "best_model.pt",
        help="Path to the model checkpoint.",
    )
    parser.add_argument("--games", type=int, default=1, help="Number of self-play games.")
    parser.add_argument(
        "--max-moves",
        type=int,
        default=200,
        help="Safety cutoff per game. Reaching this is not a natural termination.",
    )
    parser.add_argument(
        "--selection",
        choices=("greedy", "sample"),
        default="greedy",
        help="greedy matches the current GUI agent; sample explores varied moves.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature; used only with --selection sample and must be > 0.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampled play.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "pass_eval_results",
        help="Directory in which CSV and JSON results will be saved.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Inference device.",
    )
    return parser.parse_args()


def occupied_points(env: GoGameEnv) -> int:
    return sum(
        env.board.get(row, col) is not None
        for row in range(env.board_size)
        for col in range(env.board_size)
    )


def action_label(action: int) -> str:
    if action == PASS_MOVE:
        return "PASS"
    row, col = divmod(int(action), BOARD_SIZE)
    return f"row_{row + 1}_col_{col + 1}"


def choose_action(agent: TRMAgent, env: GoGameEnv, selection: str,
                  temperature: float, rng: np.random.Generator) -> tuple[int, float, float]:
    """Return action, original policy P(PASS), and probability of chosen action."""
    _, legal_mask, probabilities = agent.get_policy(env)

    # get_policy already masks illegal board moves. PASS remains legal while
    # the game is active, so the returned probabilities form a legal policy.
    legal_probs = probabilities.detach().cpu().numpy().astype(np.float64)
    pass_probability = float(legal_probs[env.pass_index])

    if selection == "greedy":
        action = int(np.argmax(legal_probs))
    else:
        # Temperature scaling of the policy distribution. Clamp only to avoid
        # log(0); zero-probability illegal actions remain effectively zero.
        log_probs = np.full_like(legal_probs, -np.inf)
        positive = legal_probs > 0
        log_probs[positive] = np.log(legal_probs[positive]) / temperature
        log_probs -= np.max(log_probs)
        sampled_probs = np.exp(log_probs)
        sampled_probs /= sampled_probs.sum()
        action = int(rng.choice(len(sampled_probs), p=sampled_probs))

    if not bool(legal_mask[action].item()):
        raise RuntimeError(f"Policy selected illegal action {action} despite legal-move masking.")

    return action, pass_probability, float(legal_probs[action])


def run_game(game_number: int, agent: TRMAgent, args: argparse.Namespace,
             rng: np.random.Generator) -> tuple[dict, list[dict]]:
    env = GoGameEnv(board_size=BOARD_SIZE, komi=KOMI)
    move_rows: list[dict] = []
    pass_moves = 0
    first_pass_move = None
    first_pass_occupied = None
    started = time.monotonic()

    for move_number in range(1, args.max_moves + 1):
        if env.game_over:
            break

        player = "Black" if env.current_player == "b" else "White"
        occupied_before = occupied_points(env)
        action, pass_probability, chosen_probability = choose_action(
            agent, env, args.selection, args.temperature, rng
        )

        is_pass = action == env.pass_index
        if is_pass:
            pass_moves += 1
            if first_pass_move is None:
                first_pass_move = move_number
                first_pass_occupied = occupied_before

        if move_number <= 10 or move_number % 10 == 0 or is_pass:
            print(
                f"[PASS-EVAL] Game {game_number}/{args.games}, move {move_number}: "
                f"{player} -> {action_label(action)}; "
                f"P(PASS)={pass_probability:.4f}; occupied={occupied_before}",
                flush=True,
            )

        # Crucial: only execute the action chosen by the model. There is no
        # automatic second PASS when the first player passes.
        game_over = env.step(action)

        move_rows.append({
            "game": game_number,
            "move_number": move_number,
            "player": player,
            "action": action_label(action),
            "action_index": action,
            "is_pass": is_pass,
            "occupied_before_move": occupied_before,
            "pass_probability": pass_probability,
            "chosen_action_probability": chosen_probability,
            "consecutive_passes_after_move": env.consecutive_passes,
            "game_over_after_move": game_over,
        })

        if game_over:
            break

    naturally_ended = bool(env.game_over)
    moves_played = len(move_rows)
    final_occupied = occupied_points(env)
    score = env.calculate_score() if naturally_ended else None

    result = {
        "game": game_number,
        "termination": "two_consecutive_model_passes" if naturally_ended else "move_limit_reached",
        "ended_naturally": naturally_ended,
        "moves_played": moves_played,
        "pass_moves": pass_moves,
        "first_pass_move": first_pass_move,
        "occupied_points_when_first_passed": first_pass_occupied,
        "occupied_points_at_stop": final_occupied,
        "winner_by_area_score": score["winner"] if score else None,
        "black_score": score["black"] if score else None,
        "white_score": score["white"] if score else None,
        "margin": score["margin"] if score else None,
        "duration_seconds": round(time.monotonic() - started, 3),
    }
    return result, move_rows


def main() -> int:
    args = parse_args()
    if args.games < 1:
        raise ValueError("--games must be at least 1.")
    if args.max_moves < 1:
        raise ValueError("--max-moves must be at least 1.")
    if args.temperature <= 0:
        raise ValueError("--temperature must be greater than 0.")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but CUDA is not available.")

    print(f"[PASS-EVAL] Checkpoint: {args.checkpoint.resolve()}", flush=True)
    print(f"[PASS-EVAL] Device: {device}", flush=True)
    print(f"Move selection: {args.selection}")
    if args.selection == "sample":
        print(f"Temperature: {args.temperature}")
    print("PASS handling: model must choose both consecutive PASS moves; no auto-pass.")

    # Load one agent and use the same weights for Black and White. The state
    # encoder adapts to the current player, including the color channel.
    print("[PASS-EVAL] Loading checkpoint and constructing TRM agent...", flush=True)
    agent = TRMAgent(checkpoint_path=str(args.checkpoint), device=device)
    print("[PASS-EVAL] Checkpoint loaded. Starting self-play...", flush=True)
    rng = np.random.default_rng(args.seed)

    game_results: list[dict] = []
    all_moves: list[dict] = []
    for game_number in range(1, args.games + 1):
        print(f"\n[PASS-EVAL] Starting game {game_number}/{args.games}...", flush=True)
        result, move_rows = run_game(game_number, agent, args, rng)
        game_results.append(result)
        all_moves.extend(move_rows)
        print(
            f"Game {game_number:02d}: {result['termination']} | "
            f"moves={result['moves_played']} | PASS={result['pass_moves']} | "
            f"occupied={result['occupied_points_at_stop']}",
            flush=True,
        )

    natural_ends = sum(result["ended_naturally"] for result in game_results)
    move_limit_games = args.games - natural_ends
    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "device": device,
        "selection": args.selection,
        "temperature": args.temperature if args.selection == "sample" else None,
        "seed": args.seed if args.selection == "sample" else None,
        "games_requested": args.games,
        "max_moves_per_game": args.max_moves,
        "natural_terminations": natural_ends,
        "natural_termination_rate": natural_ends / args.games,
        "games_stopped_by_move_limit": move_limit_games,
        "total_pass_moves": sum(result["pass_moves"] for result in game_results),
        "games": game_results,
        "interpretation": (
            "Self-play diagnostic only; it is not a measurement of playing strength or win rate."
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    games_path = args.output_dir / "games.csv"
    moves_path = args.output_dir / "moves.csv"

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
        file.write("\n")

    game_fields = list(game_results[0].keys())
    with games_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=game_fields)
        writer.writeheader()
        writer.writerows(game_results)

    move_fields = [
        "game", "move_number", "player", "action", "action_index", "is_pass",
        "occupied_before_move", "pass_probability", "chosen_action_probability",
        "consecutive_passes_after_move", "game_over_after_move",
    ]
    with moves_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=move_fields)
        writer.writeheader()
        writer.writerows(all_moves)

    print("\n=== SUMMARY ===", flush=True)
    print(f"Natural endings (two model-chosen PASSes): {natural_ends}/{args.games} ({natural_ends / args.games:.1%})", flush=True)
    print(f"Stopped by safety move limit: {move_limit_games}/{args.games}", flush=True)
    print(f"Total PASS moves predicted: {summary['total_pass_moves']}", flush=True)
    print(f"Results saved to: {args.output_dir.resolve()}", flush=True)
    print("Files: summary.json, games.csv, moves.csv", flush=True)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
