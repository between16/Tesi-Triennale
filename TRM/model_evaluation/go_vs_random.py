"""Play TRM checkpoints against a uniform-random legal-move baseline.

Example:
    python model_evaluation/go_vs_random.py --board-size 9 --games 20 --seed 123 \
        --output model_evaluation/results/9x9_go/random-baseline.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
TRM_DIR = SCRIPT_DIR.parent
if str(TRM_DIR) not in sys.path:
    sys.path.insert(0, str(TRM_DIR))

from agents import (
    DEFAULT_PASS_LOGIT_PENALTY,
    DEFAULT_PASS_PENALTY_BEFORE_STONES,
    TRMAgent,
)
from game_env import GoGameEnv
from model_evaluation.go_match import (
    DEFAULT_CHECKPOINTS,
    DEFAULT_RULES_PATH,
    load_board_rules,
)

RESULT_FIELDS = [
    "game",
    "board_size",
    "komi",
    "target_checkpoint",
    "katago_model",
    "katago_config",
    "search_max_visits",
    "pass_logit_penalty",
    "pass_penalty_before_stones",
    "rules_file",
    "opponent_type",
    "opponent",
    "target_color",
    "winner",
    "target_result",
    "termination",
    "completed",
    "moves",
    "target_pass_count",
    "target_passes_before_16_stones",
    "pass_penalty_decisions",
    "black_score",
    "white_score",
    "margin",
    "seed",
]


def random_legal_action(env: GoGameEnv, rng: np.random.Generator) -> int:
    """Choose uniformly from every currently legal action, including PASS."""
    legal_mask = env.get_legal_moves_mask().cpu().numpy()
    legal_actions = np.flatnonzero(legal_mask)
    if legal_actions.size == 0:
        raise RuntimeError("The environment returned no legal actions.")
    return int(rng.choice(legal_actions))


def play_one_game(
    game_number: int,
    board_size: int,
    komi: float,
    agent: TRMAgent,
    target_color: str,
    rng: np.random.Generator,
    max_moves: int,
    checkpoint: Path,
    rules_path: Path,
    seed: int,
) -> dict[str, Any]:
    env = GoGameEnv(board_size=board_size, komi=komi)
    target_winner: str | None = None
    termination = "move_limit"
    completed = False
    move_count = 0
    target_pass_count = 0
    target_passes_before_16_stones = 0
    pass_penalty_decisions = 0
    pass_action = board_size * board_size

    for move_count in range(1, max_moves + 1):
        if env.current_player == target_color:
            occupied_stones = sum(
                env.board.get(row, column) is not None
                for row in range(board_size)
                for column in range(board_size)
            )
            if (
                agent.pass_logit_penalty
                and occupied_stones < agent.pass_penalty_before_stones
            ):
                pass_penalty_decisions += 1
            action = agent.get_move(env)
            if action == pass_action:
                target_pass_count += 1
                if occupied_stones < 16:
                    target_passes_before_16_stones += 1
        else:
            action = random_legal_action(env, rng)

        if env.step(action):
            score = env.calculate_score()
            target_winner = {"Black": "b", "White": "w", "Draw": "draw"}[
                score["winner"]
            ]
            termination = "two_passes"
            completed = True
            break

    score = env.calculate_score() if termination == "two_passes" else None
    if target_winner is None:
        target_result = "incomplete"
        winner = ""
    elif target_winner == "draw":
        target_result = "draw"
        winner = "draw"
    else:
        target_result = "win" if target_winner == target_color else "loss"
        winner = "black" if target_winner == "b" else "white"

    return {
        "game": game_number,
        "board_size": board_size,
        "komi": komi,
        "target_checkpoint": str(checkpoint),
        "katago_model": "",
        "katago_config": "",
        "search_max_visits": "",
        "pass_logit_penalty": agent.pass_logit_penalty,
        "pass_penalty_before_stones": agent.pass_penalty_before_stones,
        "rules_file": str(rules_path),
        "opponent_type": "random",
        "opponent": "uniform_random_legal",
        "target_color": "black" if target_color == "b" else "white",
        "winner": winner,
        "target_result": target_result,
        "termination": termination,
        "completed": str(completed).lower(),
        "moves": move_count,
        "target_pass_count": target_pass_count,
        "target_passes_before_16_stones": target_passes_before_16_stones,
        "pass_penalty_decisions": pass_penalty_decisions,
        "black_score": "" if score is None else score["black"],
        "white_score": "" if score is None else score["white"],
        "margin": "" if score is None else score["margin"],
        "seed": seed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local TRM-vs-uniform-random games under custom rules."
    )
    parser.add_argument("--board-size", type=int, choices=(9, 19), required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES_PATH)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--max-moves", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--pass-logit-penalty",
        type=float,
        default=DEFAULT_PASS_LOGIT_PENALTY,
        help="Early-game PASS logit penalty (default: enabled).",
    )
    parser.add_argument(
        "--pass-penalty-before-stones",
        type=int,
        default=DEFAULT_PASS_PENALTY_BEFORE_STONES,
        help="Apply the experimental PASS penalty while fewer than this many stones are on board.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.games < 1:
        raise ValueError("--games must be at least 1.")
    if args.max_moves is not None and args.max_moves < 1:
        raise ValueError("--max-moves must be at least 1.")

    checkpoint = args.checkpoint or DEFAULT_CHECKPOINTS[args.board_size]
    if not checkpoint.is_file():
        raise FileNotFoundError(f"TRM checkpoint not found: {checkpoint}")
    komi, _ = load_board_rules(args.rules, args.board_size)

    output = args.output or (
        SCRIPT_DIR
        / "results"
        / f"{args.board_size}x{args.board_size}_go"
        / "random-baseline.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    import torch

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    else:
        device = args.device

    agent_rng = np.random.default_rng(args.seed + 1)
    random_rng = np.random.default_rng(args.seed)
    agent = TRMAgent(
        checkpoint_path=checkpoint,
        device=device,
        move_tolerance=0.0,
        rng=agent_rng,
        pass_logit_penalty=args.pass_logit_penalty,
        pass_penalty_before_stones=args.pass_penalty_before_stones,
    )
    if agent.board_size != args.board_size:
        raise ValueError(
            f"Checkpoint is {agent.board_size}x{agent.board_size}, "
            f"but --board-size is {args.board_size}."
        )

    max_moves = args.max_moves or 4 * args.board_size * args.board_size
    with output.open("w", newline="", encoding="utf-8") as result_file:
        writer = csv.DictWriter(result_file, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for game_number in range(1, args.games + 1):
            target_color = "b" if game_number % 2 == 1 else "w"
            result = play_one_game(
                game_number=game_number,
                board_size=args.board_size,
                komi=komi,
                agent=agent,
                target_color=target_color,
                rng=random_rng,
                max_moves=max_moves,
                checkpoint=checkpoint,
                rules_path=args.rules,
                seed=args.seed,
            )
            writer.writerow(result)
            result_file.flush()
            print(
                f"Game {game_number}/{args.games}: {result['target_result']} "
                f"({result['termination']}, {result['moves']} moves)"
            )
    print(f"Match results: {output}")


if __name__ == "__main__":
    main()
