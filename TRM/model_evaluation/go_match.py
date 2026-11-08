"""Play TRM checkpoints against a local KataGo GTP engine.

Example:
    python model_evaluation/go_match.py --board-size 9 --katago-model /models/kata.bin \
        --katago-config /models/gtp.cfg --games 20 --output matches.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import queue
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np
import torch

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

DEFAULT_RULES_PATH = SCRIPT_DIR / "config" / "rules.json"
DEFAULT_CHECKPOINTS = {
    9: TRM_DIR / "checkpoints" / "best_model.pt",
    19: TRM_DIR / "checkpoints" / "19x19_go" / "best_model.pt",
}
GTP_COLUMNS = "ABCDEFGHJKLMNOPQRSTUVWXYZ"
MATCH_FIELDS = [
    "game",
    "board_size",
    "komi",
    "target_checkpoint",
    "katago_model",
    "katago_config",
    "search_max_visits",
    "human_model",
    "opponent_profile",
    "human_temperature_mode",
    "opponent_type",
    "pass_logit_penalty",
    "pass_penalty_before_stones",
    "rules_file",
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


def load_board_rules(path: Path, board_size: int) -> tuple[float, dict[str, Any]]:
    """Load rules and reject settings that do not match GoGameEnv."""
    if board_size not in (9, 19):
        raise ValueError(f"Only 9x9 and 19x19 are supported, got {board_size}.")
    with path.open(encoding="utf-8") as rules_file:
        config = json.load(rules_file)

    board = config.get("boards", {}).get(str(board_size))
    rules = config.get("rules")
    if not isinstance(board, dict) or board.get("board_size") != board_size:
        raise ValueError(f"No valid rules entry for {board_size}x{board_size} in {path}.")
    if not isinstance(rules, dict):
        raise ValueError(f"Missing rules object in {path}.")

    expected = {
        "scoring": "AREA",
        "ko": "POSITIONAL",
        "suicide": False,
        "tax": "NONE",
        "hasButton": False,
        "whiteHandicapBonus": "0",
        "friendlyPassOk": True,
        "pass_allowed": True,
        "game_end": "two_consecutive_passes",
    }
    if rules != expected:
        raise ValueError(
            "The rules configuration must match GoGameEnv exactly. "
            f"Expected {expected}, received {rules}."
        )

    komi = board.get("komi")
    if isinstance(komi, bool) or not isinstance(komi, (int, float)) or not math.isfinite(komi):
        raise ValueError(f"Invalid komi for {board_size}x{board_size}: {komi!r}.")

    gtp_rules = {
        key: rules[key]
        for key in (
            "ko",
            "scoring",
            "suicide",
            "tax",
            "hasButton",
            "whiteHandicapBonus",
            "friendlyPassOk",
        )
    }
    return float(komi), gtp_rules


def action_to_gtp(action: int, board_size: int) -> str:
    """Convert the TRM's row-major action index to a GTP coordinate."""
    if action == board_size * board_size:
        return "pass"
    if action < 0 or action >= board_size * board_size:
        raise ValueError(f"Action {action} is outside a {board_size}x{board_size} board.")
    row, col = divmod(action, board_size)
    if col >= len(GTP_COLUMNS):
        raise ValueError(f"Board size {board_size} exceeds supported GTP coordinates.")
    return f"{GTP_COLUMNS[col]}{board_size - row}"


def gtp_to_action(move: str, board_size: int) -> int:
    """Convert a GTP coordinate or pass to the TRM's row-major action index."""
    move = move.strip().upper()
    if move == "PASS":
        return board_size * board_size
    if len(move) < 2 or move[0] not in GTP_COLUMNS:
        raise ValueError(f"Invalid GTP move: {move!r}.")
    column = GTP_COLUMNS.index(move[0])
    try:
        gtp_row = int(move[1:])
    except ValueError as exc:
        raise ValueError(f"Invalid GTP move: {move!r}.") from exc
    if column >= board_size or not 1 <= gtp_row <= board_size:
        raise ValueError(f"GTP move {move!r} is outside a {board_size}x{board_size} board.")
    row = board_size - gtp_row
    return row * board_size + column


class GTPClient:
    """Small synchronous GTP client with response timeouts and error checking."""

    def __init__(self, command: list[str], timeout: float) -> None:
        self.timeout = timeout
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
        )
        if self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError("Could not open KataGo's GTP input/output pipes.")
        self.responses: queue.Queue[str | None] = queue.Queue()
        self.reader = threading.Thread(target=self._read_output, daemon=True)
        self.reader.start()

    def _read_output(self) -> None:
        assert self.process.stdout is not None
        for line in self.process.stdout:
            self.responses.put(line)
        self.responses.put(None)

    def command(self, command: str) -> str:
        if self.process.poll() is not None:
            raise RuntimeError(f"KataGo exited with status {self.process.returncode}.")
        assert self.process.stdin is not None
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()

        response_lines: list[str] = []
        while True:
            try:
                line = self.responses.get(timeout=self.timeout)
            except queue.Empty as exc:
                raise TimeoutError(
                    f"KataGo did not respond within {self.timeout:g}s to {command!r}."
                ) from exc
            if line is None:
                raise RuntimeError(
                    f"KataGo closed its GTP output while processing {command!r}."
                )
            if not line.strip():
                if response_lines:
                    break
                continue
            response_lines.append(line.rstrip("\r\n"))

        header = response_lines[0]
        if not header.startswith("="):
            raise RuntimeError(f"KataGo rejected {command!r}: {' '.join(response_lines)}")
        first_line = header[1:].strip()
        return "\n".join([first_line, *(line.strip() for line in response_lines[1:])]).strip()

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            self.process.wait(timeout=10)
        if self.process.stdin is not None:
            self.process.stdin.close()
        if self.process.stdout is not None:
            self.process.stdout.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local TRM-vs-KataGo games under the repository's custom rules."
    )
    parser.add_argument("--board-size", type=int, choices=(9, 19), required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES_PATH)
    parser.add_argument("--katago", default="katago", help="Local KataGo executable.")
    parser.add_argument("--katago-config", type=Path, required=True)
    parser.add_argument("--katago-model", type=Path, required=True)
    parser.add_argument(
        "--katago-human-model",
        type=Path,
        default=None,
        help="Optional official HumanSL network for rank-profiled weaker opponents.",
    )
    parser.add_argument(
        "--human-profile",
        choices=(
            "rank_20k", "rank_15k", "rank_10k", "rank_5k", "rank_1k",
            "rank_1d", "rank_3d", "rank_5d", "rank_9d",
        ),
        default=None,
        help="HumanSL profile; requires --katago-human-model.",
    )
    parser.add_argument(
        "--human-temperature-mode",
        choices=("full",),
        default=None,
        help=(
            "Use KataGo's official HumanSL full-range sampling settings "
            "(early=0.85, later=0.70, halflife=80, below-prob=0.01)."
        ),
    )
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--max-moves", type=int, default=None)
    parser.add_argument(
        "--max-visits",
        type=int,
        default=128,
        help="Fixed KataGo search visits per move; repeat evaluation at several budgets.",
    )
    parser.add_argument("--gtp-timeout", type=float, default=600.0)
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
        help="Apply the PASS penalty while fewer than this many stones are on board.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def configure_katago(
    client: GTPClient, board_size: int, komi: float, gtp_rules: dict[str, Any]
) -> None:
    client.command(f"boardsize {board_size}")
    client.command("clear_board")
    client.command(f"komi {komi:g}")
    client.command(f"kata-set-rules {json.dumps(gtp_rules, separators=(',', ':'))}")

    actual_rules = json.loads(client.command("kata-get-rules"))
    if any(actual_rules.get(key) != value for key, value in gtp_rules.items()):
        raise RuntimeError(
            "KataGo reported rules that differ from the requested rules: "
            f"requested {gtp_rules}, received {actual_rules}."
        )
    actual_komi = float(client.command("get_komi"))
    if not math.isclose(actual_komi, komi, abs_tol=1e-9):
        raise RuntimeError(f"KataGo komi mismatch: expected {komi}, received {actual_komi}.")


def play_one_game(
    game_number: int,
    board_size: int,
    komi: float,
    agent: TRMAgent,
    client: GTPClient,
    target_color: str,
    max_moves: int,
    seed: int,
    checkpoint: Path,
    katago_model: Path,
    katago_config: Path,
    max_visits: int,
    rules_path: Path,
    human_model: Path | None,
    human_profile: str | None,
    human_temperature_mode: str | None,
) -> dict[str, Any]:
    client.command("clear_board")
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
        color = env.current_player
        if color == target_color:
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
            client.command(f"play {color} {action_to_gtp(action, board_size)}")
        else:
            move = client.command(f"genmove {color}").strip()
            if move.lower() == "resign":
                target_winner = target_color
                termination = "katago_resign"
                completed = True
                break
            action = gtp_to_action(move, board_size)
            legal, reason = env.is_legal_move(action)
            if not legal:
                raise RuntimeError(
                    f"KataGo returned illegal move {move!r} in game {game_number}: {reason}."
                )

        game_over = env.step(action)
        if game_over:
            score = env.calculate_score()
            target_winner = "b" if score["winner"] == "Black" else "w"
            termination = "two_passes"
            completed = True
            break

    score = env.calculate_score() if termination == "two_passes" else None
    if target_winner is None:
        target_result = "incomplete"
        winner = ""
    elif score is not None and score["winner"] == "Draw":
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
        "katago_model": str(katago_model),
        "katago_config": str(katago_config),
        "search_max_visits": max_visits,
        "human_model": "" if human_model is None else str(human_model),
        "opponent_profile": "" if human_profile is None else human_profile,
        "human_temperature_mode": human_temperature_mode or "engine_default",
        "opponent_type": "human_sl" if human_model is not None else "kata1",
        "pass_logit_penalty": agent.pass_logit_penalty,
        "pass_penalty_before_stones": agent.pass_penalty_before_stones,
        "rules_file": str(rules_path),
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


def main() -> None:
    args = parse_args()
    if args.games < 1:
        raise ValueError("--games must be at least 1.")
    if args.max_visits < 1:
        raise ValueError("--max-visits must be at least 1.")
    if args.katago_human_model is not None and not args.katago_human_model.is_file():
        raise FileNotFoundError(f"KataGo HumanSL model not found: {args.katago_human_model}")
    if args.human_profile is not None and args.katago_human_model is None:
        raise ValueError("--human-profile requires --katago-human-model.")
    if args.human_temperature_mode is not None and args.katago_human_model is None:
        raise ValueError("--human-temperature-mode requires --katago-human-model.")
    human_profile = args.human_profile
    if args.katago_human_model is not None and human_profile is None:
        human_profile = "rank_20k"
    if args.max_moves is not None and args.max_moves < 1:
        raise ValueError("--max-moves must be at least 1.")
    if args.gtp_timeout <= 0 or not math.isfinite(args.gtp_timeout):
        raise ValueError("--gtp-timeout must be a positive finite number.")
    if not args.katago_config.is_file() or not args.katago_model.is_file():
        raise FileNotFoundError("KataGo config and model files must both exist locally.")
    if shutil.which(args.katago) is None and not Path(args.katago).is_file():
        raise FileNotFoundError(f"KataGo executable not found: {args.katago}")

    checkpoint = args.checkpoint or DEFAULT_CHECKPOINTS[args.board_size]
    if not checkpoint.is_file():
        raise FileNotFoundError(f"TRM checkpoint not found: {checkpoint}")
    if not math.isfinite(args.pass_logit_penalty) or args.pass_logit_penalty < 0:
        raise ValueError("--pass-logit-penalty must be finite and non-negative.")
    if args.pass_penalty_before_stones < 0:
        raise ValueError("--pass-penalty-before-stones cannot be negative.")
    if (args.pass_logit_penalty == 0) != (args.pass_penalty_before_stones == 0):
        raise ValueError(
            "Set both PASS penalty options to zero to disable the adjustment, "
            "or set both to positive values."
        )
    komi, gtp_rules = load_board_rules(args.rules, args.board_size)
    output = args.output or (
        SCRIPT_DIR
        / "results"
        / f"{args.board_size}x{args.board_size}_go"
        / "matches.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    rng = np.random.default_rng(args.seed)
    agent = TRMAgent(
        checkpoint_path=checkpoint,
        device=device,
        move_tolerance=0.0,
        rng=rng,
        pass_logit_penalty=args.pass_logit_penalty,
        pass_penalty_before_stones=args.pass_penalty_before_stones,
    )
    if agent.board_size != args.board_size:
        raise ValueError(
            f"Checkpoint is {agent.board_size}x{agent.board_size}, "
            f"but --board-size is {args.board_size}."
        )

    command = [
        args.katago,
        "gtp",
        "-config",
        str(args.katago_config),
        "-model",
        str(args.katago_model),
        "-override-config",
        (
            f"maxVisits={args.max_visits},numSearchThreads=1,"
            f"allowResignation=false,logDir={SCRIPT_DIR / 'katago-local' / 'logs'},"
            "delayMoveScale=0,delayMoveMax=0"
        ),
    ]
    if args.katago_human_model is not None:
        command.extend(["-human-model", str(args.katago_human_model)])
        command[command.index("-override-config") + 1] += (
            f",humanSLProfile={human_profile},humanSLChosenMoveProp=1.0,"
            "humanSLChosenMoveIgnorePass=true,"
            "humanSLChosenMovePiklLambda=100000000"
        )
        if args.human_temperature_mode == "full":
            command[command.index("-override-config") + 1] += (
                ",chosenMoveTemperatureEarly=0.85,"
                "chosenMoveTemperature=0.70,"
                "chosenMoveTemperatureHalflife=80,"
                "chosenMoveTemperatureOnlyBelowProb=0.01,"
                "chosenMoveSubtract=0,chosenMovePrune=0"
            )
    client = GTPClient(command, timeout=args.gtp_timeout)
    max_moves = args.max_moves or 4 * args.board_size * args.board_size
    try:
        configure_katago(client, args.board_size, komi, gtp_rules)
        with output.open("w", newline="", encoding="utf-8") as result_file:
            writer = csv.DictWriter(
                result_file, fieldnames=MATCH_FIELDS, lineterminator="\n"
            )
            writer.writeheader()
            for game_number in range(1, args.games + 1):
                target_color = "b" if game_number % 2 == 1 else "w"
                result = play_one_game(
                    game_number,
                    args.board_size,
                    komi,
                    agent,
                    client,
                    target_color,
                    max_moves,
                    args.seed,
                    checkpoint,
                    args.katago_model,
                    args.katago_config,
                    args.max_visits,
                    args.rules,
                    args.katago_human_model,
                    human_profile,
                    args.human_temperature_mode,
                )
                writer.writerow(result)
                result_file.flush()
                print(
                    f"Game {game_number}/{args.games}: {result['target_result']} "
                    f"({result['termination']}, {result['moves']} moves)"
                )
        print(f"Match results: {output}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
