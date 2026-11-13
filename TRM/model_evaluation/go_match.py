"""Play TRM checkpoints against a local KataGo GTP engine.

Example:
    python model_evaluation/go_match.py --board-size 9 --katago-model /models/kata.bin \
        --katago-config /models/gtp.cfg --games 20

By default, the CSV and JSON full report are saved in:
    model_evaluation/results/<board>x<board>_go/full_reports/

The standalone Markdown summary is saved in:
    model_evaluation/results/<board>x<board>_go/summaries/
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import queue
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
TRM_DIR = SCRIPT_DIR.parent
REPO_ROOT = TRM_DIR.parent
if str(TRM_DIR) not in sys.path:
    sys.path.insert(0, str(TRM_DIR))

from agents import (
    DEFAULT_PASS_LOGIT_PENALTY,
    DEFAULT_PASS_PENALTY_BEFORE_STONES,
    TRMAgent,
)
from game_env import GoGameEnv
from negamax import NegamaxAgent

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
    "search_mode",
    "search_depth",
    "search_candidates",
    "search_nodes",
    "search_leaves",
    "search_cutoffs",
    "search_policy_calls",
    "search_time_seconds",
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
    "target_score_margin",
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

    def __init__(
        self,
        command: list[str],
        timeout: float,
        environment: dict[str, str] | None = None,
    ) -> None:
        self.timeout = timeout
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
            env=environment,
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
        "--search",
        choices=("policy", "negamax"),
        default="policy",
        help="Target move selection: direct TRM policy or TRM-guided Negamax.",
    )
    parser.add_argument(
        "--search-depth",
        type=int,
        default=2,
        help="Negamax depth in plies (default: 2; ignored by --search policy).",
    )
    parser.add_argument(
        "--search-candidates",
        type=int,
        default=None,
        help=(
            "Policy-ranked board moves per node (defaults to 5 on 9x9 and "
            "3 on 19x19; ignored by --search policy)."
        ),
    )
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
    parser.add_argument(
        "--report-output",
        type=Path,
        default=None,
        help="Optional JSON full-report path; generated automatically by default.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Optional Markdown summary path; generated automatically by default.",
    )
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


def _file_metadata(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": _portable_path(path),
        "size_bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _portable_path(path: Path | str) -> str:
    """Represent repository files relative to the repository, never this host."""
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    try:
        return candidate.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return Path(path).name if Path(path).is_absolute() else Path(path).as_posix()


def _portable_argument(argument: str) -> str:
    if Path(argument).is_absolute():
        return _portable_path(argument)
    if "=" in argument:
        key, value = argument.split("=", 1)
        if Path(value).is_absolute():
            return f"{key}={_portable_path(value)}"
    return argument


def _portable_invocation(arguments: list[str]) -> str:
    portable_arguments = [
        "python3" if index == 0 else _portable_argument(argument)
        for index, argument in enumerate(arguments)
    ]
    return shlex.join(portable_arguments)


def _katago_runtime_environment() -> tuple[dict[str, str], tempfile.TemporaryDirectory | None]:
    environment = os.environ.copy()
    library_paths: list[str] = []
    bundled_libraries = (
        SCRIPT_DIR / "katago-local" / "engine" / "squashfs-root" / "usr" / "lib"
    )
    if bundled_libraries.is_dir():
        library_paths.append(str(bundled_libraries))

    cudnn_wheel = SCRIPT_DIR / "katago-local" / "downloads" / "nvidia_cudnn_cu13-9.28.1.4.whl"
    temporary_directory = None
    if cudnn_wheel.is_file():
        temporary_directory = tempfile.TemporaryDirectory(prefix="katago-cudnn-")
        with zipfile.ZipFile(cudnn_wheel) as wheel:
            for name in wheel.namelist():
                if name.startswith("nvidia/cudnn/lib/") and ".so" in name:
                    destination = Path(temporary_directory.name, Path(name).name)
                    with wheel.open(name) as source, destination.open("wb") as target:
                        shutil.copyfileobj(source, target)
        library_paths.insert(0, temporary_directory.name)

    existing_library_path = environment.get("LD_LIBRARY_PATH", "")
    if existing_library_path:
        library_paths.append(existing_library_path)
    if library_paths:
        environment["LD_LIBRARY_PATH"] = ":".join(library_paths)
    return environment, temporary_directory


def _unique_path(path: Path) -> Path:
    """Keep prior runs intact by adding a UTC timestamp when a path exists."""
    if not path.exists():
        return path
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    candidate = path.with_name(f"{path.stem}.run-{timestamp}{path.suffix}")
    suffix = 1
    while candidate.exists():
        candidate = path.with_name(
            f"{path.stem}.run-{timestamp}-{suffix}{path.suffix}"
        )
        suffix += 1
    return candidate


def _wilson_interval(successes: int, total: int) -> list[float] | None:
    if total == 0:
        return None
    z = 1.959963984540054
    rate = successes / total
    denominator = 1 + z * z / total
    centre = (rate + z * z / (2 * total)) / denominator
    radius = (
        z
        * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total))
        / denominator
    )
    return [max(0.0, centre - radius), min(1.0, centre + radius)]


def _match_summary(
    results: list[dict[str, Any]],
    requested_games: int | None = None,
) -> dict[str, Any]:
    if requested_games is None:
        requested_games = len(results)
    completed = [row for row in results if row["completed"] == "true"]
    wins = sum(row["target_result"] == "win" for row in completed)
    draws = sum(row["target_result"] == "draw" for row in completed)
    losses = sum(row["target_result"] == "loss" for row in completed)
    scored = [
        float(row["target_score_margin"])
        for row in completed
        if row["target_score_margin"] != ""
    ]
    target_color_summary: dict[str, dict[str, Any]] = {}
    for color in ("black", "white"):
        color_rows = [row for row in results if row["target_color"] == color]
        completed_color_rows = [
            row for row in color_rows if row["completed"] == "true"
        ]
        color_gaps = [
            float(row["target_score_margin"])
            for row in completed_color_rows
            if row["target_score_margin"] != ""
        ]
        color_wins = sum(
            row["target_result"] == "win" for row in completed_color_rows
        )
        color_draws = sum(
            row["target_result"] == "draw" for row in completed_color_rows
        )
        color_losses = sum(
            row["target_result"] == "loss" for row in completed_color_rows
        )
        target_color_summary[color] = {
            "games_played": len(color_rows),
            "completed_games": len(completed_color_rows),
            "incomplete_games": len(color_rows) - len(completed_color_rows),
            "wins": color_wins,
            "draws": color_draws,
            "losses": color_losses,
            "win_rate": (
                color_wins / len(completed_color_rows)
                if completed_color_rows else None
            ),
            "mean_target_score_margin": (
                sum(color_gaps) / len(color_gaps) if color_gaps else None
            ),
            "scored_games": len(color_gaps),
        }
    return {
        "requested_games": requested_games,
        "games_recorded": len(results),
        "not_started_games": max(0, requested_games - len(results)),
        "completed_games": len(completed),
        "incomplete_games": len(results) - len(completed),
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "win_rate": wins / len(completed) if completed else None,
        "score_rate": (wins + 0.5 * draws) / len(completed) if completed else None,
        "win_rate_95ci_wilson": _wilson_interval(wins, len(completed)),
        "target_color_results": target_color_summary,
        "termination_counts": {
            reason: sum(row["termination"] == reason for row in results)
            for reason in sorted({row["termination"] for row in results})
        },
        "mean_moves": (
            sum(int(row["moves"]) for row in results) / len(results)
            if results else None
        ),
        "mean_target_score_margin": (
            sum(scored) / len(scored) if scored else None
        ),
        "median_target_score_margin": (
            float(np.median(scored)) if scored else None
        ),
        "mean_absolute_target_score_margin": (
            sum(abs(value) for value in scored) / len(scored) if scored else None
        ),
        "positive_gap_games": sum(value > 0 for value in scored),
        "negative_gap_games": sum(value < 0 for value in scored),
        "zero_gap_games": sum(value == 0 for value in scored),
        "scored_games": len(scored),
        "target_passes": sum(int(row["target_pass_count"]) for row in results),
        "target_passes_before_16_stones": sum(
            int(row["target_passes_before_16_stones"]) for row in results
        ),
        "pass_penalty_decisions": sum(
            int(row["pass_penalty_decisions"]) for row in results
        ),
        "search": {
            "nodes": sum(int(row["search_nodes"]) for row in results),
            "leaves": sum(int(row["search_leaves"]) for row in results),
            "cutoffs": sum(int(row["search_cutoffs"]) for row in results),
            "policy_calls": sum(
                int(row["search_policy_calls"]) for row in results
            ),
            "elapsed_seconds": sum(
                float(row["search_time_seconds"]) for row in results
            ),
        },
    }


def _comparison_key(parameters: dict[str, Any], artifact_hashes: dict[str, Any]) -> str:
    comparable_parameters = {
        key: value
        for key, value in parameters.items()
        if key not in {
            "search",
            "search_depth",
            "search_candidates",
            "effective_search_depth",
            "output",
            "requested_output",
            "report_output",
            "summary_output",
            "effective_search_candidates",
        }
    }
    payload = {
        "parameters": comparable_parameters,
        "artifacts": artifact_hashes,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_report(
    report_path: Path,
    *,
    status: str,
    error: str | None,
    started_at: datetime,
    elapsed_seconds: float,
    invocation: str,
    parameters: dict[str, Any],
    engine_command: list[str],
    engine_name: str,
    engine_version: str,
    artifacts: dict[str, Any],
    results: list[dict[str, Any]],
    csv_path: Path,
    source_revision: str,
    source_dirty: bool,
) -> None:
    artifact_hashes = {
        name: metadata["sha256"] if metadata is not None else None
        for name, metadata in artifacts.items()
    }
    report = {
        "schema_version": 1,
        "status": status,
        "error": error,
        "started_at_utc": started_at.isoformat(),
        "duration_seconds": elapsed_seconds,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_revision": source_revision,
        "source_dirty": source_dirty,
        "invocation": {
            "command": invocation,
            "working_directory": _portable_path(Path.cwd()),
            "stored_path_base": "repository root",
            "python": sys.version,
        },
        "runtime": {
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device": parameters.get("effective_device"),
            "gpu": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
        },
        "parameters": parameters,
        "comparison_key": _comparison_key(parameters, artifact_hashes),
        "artifacts": artifacts,
        "engine": {
            "name": engine_name,
            "version": engine_version,
            "command": engine_command,
        },
        "results_csv": {
            "path": _portable_path(csv_path),
            "sha256": _file_metadata(csv_path)["sha256"] if csv_path.is_file() else None,
        },
        "summary": _match_summary(
            results,
            int(parameters.get("games", len(results))),
        ),
        "games": results,
    }
    with report_path.open("x", encoding="utf-8") as file:
        json.dump(report, file, indent=2, allow_nan=False)
        file.write("\n")



def _markdown_cell(value: Any) -> str:
    """Render a value safely inside a Markdown table cell."""
    if value is None or value == "":
        return "—"
    if isinstance(value, bool):
        rendered = "yes" if value else "no"
    elif isinstance(value, (dict, list, tuple)):
        rendered = json.dumps(value, ensure_ascii=False, sort_keys=True)
    else:
        rendered = str(value)
    return rendered.replace("|", r"\|").replace("\r", " ").replace("\n", " ")


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(_markdown_cell(item) for item in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(_markdown_cell(item) for item in row) + " |"
        for row in rows
    )
    return lines


def _format_number(value: Any, digits: int = 2, suffix: str = "") -> str:
    if value is None or value == "":
        return "—"
    return f"{float(value):,.{digits}f}{suffix}"


def _format_percent(value: Any) -> str:
    if value is None:
        return "—"
    return f"{float(value) * 100:.1f}%"


def _format_gap(value: Any) -> str:
    if value is None or value == "":
        return "—"
    return f"{float(value):+.1f} pt"


def _write_summary_report(
    summary_path: Path,
    *,
    status: str,
    error: str | None,
    started_at: datetime,
    elapsed_seconds: float,
    invocation: str,
    parameters: dict[str, Any],
    engine_name: str,
    engine_version: str,
    results: list[dict[str, Any]],
    csv_path: Path,
    report_path: Path,
) -> None:
    """Write a standalone, human-readable summary of the KataGo match."""
    requested_games = int(parameters.get("games", len(results)))
    summary = _match_summary(results, requested_games)
    board_size = parameters.get("board_size", "?")
    search_mode = parameters.get("search", "unknown")
    finished_at = datetime.now(timezone.utc)
    status_labels = {"completed": "COMPLETATO", "failed": "FALLITO"}

    lines = [
        f"# Riepilogo test TRM vs KataGo — {board_size}x{board_size}",
        "",
        f"- **Stato:** {status_labels.get(status, status.upper())}",
        f"- **Inizio (UTC):** {started_at.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **Fine (UTC):** {finished_at.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **Durata:** {elapsed_seconds / 60:.1f} minuti ({elapsed_seconds:.1f} secondi)",
        f"- **Avversario:** {engine_name} {engine_version}",
        f"- **Modalità TRM:** {search_mode}",
        "",
        "## Parametri del test",
        "",
        "Comando eseguito:",
        "```bash",
        invocation,
        "```",
        "",
        "La tabella riporta gli argomenti CLI (compresi i valori di default) e i parametri effettivi risolti dallo script.",
        "",
    ]

    parameter_rows: list[list[Any]] = []
    cli_keys = list(parameters.keys())
    for key in cli_keys:
        if key in {
            "requested_output",
            "komi",
            "effective_gtp_rules",
            "effective_max_moves",
            "effective_human_profile",
            "effective_device",
            "effective_search_candidates",
            "effective_search_depth",
        }:
            continue
        parameter_rows.append([f"--{key.replace('_', '-')}", parameters[key]])
    lines.extend(_markdown_table(["Argomento CLI", "Valore"], parameter_rows))
    lines.extend(
        [
            "",
            "### Parametri effettivi e file generati",
            "",
        ]
    )
    effective_rows = [
        ["Checkpoint TRM", parameters.get("checkpoint")],
        ["Device effettivo", parameters.get("effective_device")],
        ["Komi", parameters.get("komi")],
        ["Limite mosse effettivo", parameters.get("effective_max_moves")],
        ["Regole GTP effettive", parameters.get("effective_gtp_rules")],
        ["Profilo HumanSL effettivo", parameters.get("effective_human_profile")],
        ["Candidati Negamax effettivi", parameters.get("effective_search_candidates")],
        ["Profondità Negamax effettiva", parameters.get("effective_search_depth")],
        ["CSV partite", _portable_path(csv_path)],
        ["Full report JSON", _portable_path(report_path)],
        ["Riepilogo Markdown", _portable_path(summary_path)],
    ]
    lines.extend(_markdown_table(["Parametro", "Valore"], effective_rows))

    lines.extend(["", "## Risultati complessivi", ""])
    completed = summary["completed_games"]
    win_interval = summary["win_rate_95ci_wilson"]
    interval_text = (
        f"{win_interval[0] * 100:.1f}%–{win_interval[1] * 100:.1f}%"
        if win_interval is not None else "—"
    )
    termination_text = (
        ", ".join(
            f"{reason}: {count}"
            for reason, count in summary["termination_counts"].items()
        )
        or "—"
    )
    aggregate_rows = [
        ["Partite richieste", summary["requested_games"]],
        ["Partite registrate", summary["games_recorded"]],
        ["Partite completate", completed],
        ["Partite incomplete", summary["incomplete_games"]],
        ["Partite non iniziate", summary["not_started_games"]],
        ["Vittorie TRM", summary["wins"]],
        ["Pareggi", summary["draws"]],
        ["Sconfitte TRM (vittorie KataGo)", summary["losses"]],
        ["Win rate TRM", _format_percent(summary["win_rate"])],
        ["Score rate TRM (vittoria=1, pareggio=0.5)", _format_percent(summary["score_rate"])],
        ["Intervallo 95% win rate (Wilson)", interval_text],
        ["Gap medio TRM", _format_gap(summary["mean_target_score_margin"])],
        ["Gap mediano TRM", _format_gap(summary["median_target_score_margin"])],
        ["Gap medio assoluto", _format_number(summary["mean_absolute_target_score_margin"], 1, " pt")],
        ["Partite con punteggio numerico", summary["scored_games"]],
        ["Gap a favore TRM / a favore KataGo / pari", (
            f"{summary['positive_gap_games']} / {summary['negative_gap_games']} / {summary['zero_gap_games']}"
        )],
        ["Mosse medie per partita registrata", _format_number(summary["mean_moves"], 1)],
        ["PASS giocati dal TRM", summary["target_passes"]],
        ["PASS prima di 16 pietre", summary["target_passes_before_16_stones"]],
        ["Decisioni con penalità PASS attiva", summary["pass_penalty_decisions"]],
        ["Terminazioni", termination_text],
    ]
    lines.extend(_markdown_table(["Indicatore", "Risultato"], aggregate_rows))
    lines.extend(
        [
            "",
            "*Il gap è calcolato dal punto di vista del TRM: un valore positivo indica un vantaggio del TRM; un valore negativo indica un vantaggio di KataGo. Il gap è disponibile solo per le partite terminate con un punteggio numerico.*",
            "",
            "## Risultati per colore del TRM",
            "",
        ]
    )
    color_rows = []
    for color, label in (("black", "Nero"), ("white", "Bianco")):
        color_summary = summary["target_color_results"][color]
        color_rows.append(
            [
                label,
                color_summary["games_played"],
                color_summary["completed_games"],
                f"{color_summary['wins']}-{color_summary['draws']}-{color_summary['losses']}",
                _format_percent(color_summary["win_rate"]),
                _format_gap(color_summary["mean_target_score_margin"]),
            ]
        )
    lines.extend(
        _markdown_table(
            ["Colore TRM", "Giocate", "Completate", "V-P-S", "Win rate", "Gap medio"],
            color_rows,
        )
    )

    search = summary["search"]
    lines.extend(
        [
            "",
            "## Statistiche di ricerca",
            "",
        ]
    )
    search_rows = [
        ["Nodi esplorati", search["nodes"]],
        ["Foglie valutate", search["leaves"]],
        ["Cutoff", search["cutoffs"]],
        ["Chiamate alla policy", search["policy_calls"]],
        ["Tempo totale ricerca", f"{search['elapsed_seconds']:.2f} s"],
    ]
    lines.extend(_markdown_table(["Indicatore", "Totale"], search_rows))
    lines.append("")
    lines.append("*Le statistiche di ricerca sono specifiche del Negamax; con `--search policy` restano a zero.*")

    lines.extend(["", "## Risultato di ogni partita", ""])
    game_rows: list[list[Any]] = []
    result_labels = {
        "win": "Vittoria",
        "draw": "Pareggio",
        "loss": "Sconfitta",
        "incomplete": "Incompleta",
    }
    color_labels = {"black": "Nero", "white": "Bianco"}
    winner_labels = {"black": "Nero", "white": "Bianco", "draw": "Pareggio"}
    termination_labels = {
        "two_passes": "Due pass consecutivi",
        "move_limit": "Limite mosse",
        "katago_resign": "Resa KataGo",
    }
    for row in results:
        black_score = row["black_score"]
        white_score = row["white_score"]
        score_text = (
            f"{black_score}–{white_score}"
            if black_score != "" and white_score != ""
            else "—"
        )
        game_rows.append(
            [
                row["game"],
                color_labels.get(row["target_color"], row["target_color"]),
                result_labels.get(row["target_result"], row["target_result"]),
                winner_labels.get(row["winner"], row["winner"] or "—"),
                termination_labels.get(row["termination"], row["termination"]),
                row["moves"],
                score_text,
                _format_gap(row["target_score_margin"]),
            ]
        )
    if game_rows:
        lines.extend(
            _markdown_table(
                ["#", "Colore TRM", "Esito TRM", "Vincitore", "Terminazione", "Mosse", "Score B–W", "Gap TRM"],
                game_rows,
            )
        )
    else:
        lines.append("Nessuna partita è stata completata o registrata.")

    if error:
        lines.extend(["", "## Errore", "", "```text", error, "```"])
    lines.extend(
        [
            "",
            "## Altri file",
            "",
            f"- CSV dettagliato: `{_portable_path(csv_path)}`",
            f"- Full report JSON: `{_portable_path(report_path)}`",
            "",
        ]
    )
    with summary_path.open("x", encoding="utf-8") as file:
        file.write("\n".join(lines))
        file.write("\n")

def make_agent(
    policy_agent: TRMAgent,
    search: str,
    search_depth: int,
    search_candidates: int,
) -> TRMAgent | NegamaxAgent:
    if search == "negamax":
        return NegamaxAgent(
            policy_agent=policy_agent,
            depth=search_depth,
            max_candidates=search_candidates,
        )
    return policy_agent


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
    policy_agent = agent.policy_agent if isinstance(agent, NegamaxAgent) else agent
    search_mode = "negamax" if isinstance(agent, NegamaxAgent) else "policy"
    search_depth = agent.depth if isinstance(agent, NegamaxAgent) else 0
    search_candidates = agent.max_candidates if isinstance(agent, NegamaxAgent) else 0
    target_winner: str | None = None
    termination = "move_limit"
    completed = False
    move_count = 0
    target_pass_count = 0
    target_passes_before_16_stones = 0
    pass_penalty_decisions = 0
    search_nodes = 0
    search_leaves = 0
    search_cutoffs = 0
    search_policy_calls = 0
    search_time_seconds = 0.0
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
                getattr(policy_agent, "pass_logit_penalty", 0)
                and occupied_stones
                < getattr(policy_agent, "pass_penalty_before_stones", 0)
            ):
                pass_penalty_decisions += 1
            action = agent.get_move(env)
            if isinstance(agent, NegamaxAgent):
                stats = agent.last_search_stats
                search_nodes += stats.nodes
                search_leaves += stats.leaves
                search_cutoffs += stats.cutoffs
                search_policy_calls += stats.policy_calls
                search_time_seconds += stats.elapsed_seconds
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
        "target_checkpoint": _portable_path(checkpoint),
        "katago_model": _portable_path(katago_model),
        "katago_config": _portable_path(katago_config),
        "search_max_visits": max_visits,
        "human_model": "" if human_model is None else _portable_path(human_model),
        "opponent_profile": "" if human_profile is None else human_profile,
        "human_temperature_mode": human_temperature_mode or "engine_default",
        "opponent_type": "human_sl" if human_model is not None else "kata1",
        "search_mode": search_mode,
        "search_depth": search_depth,
        "search_candidates": search_candidates,
        "search_nodes": search_nodes,
        "search_leaves": search_leaves,
        "search_cutoffs": search_cutoffs,
        "search_policy_calls": search_policy_calls,
        "search_time_seconds": search_time_seconds,
        "pass_logit_penalty": getattr(policy_agent, "pass_logit_penalty", 0),
        "pass_penalty_before_stones": getattr(
            policy_agent, "pass_penalty_before_stones", 0
        ),
        "rules_file": _portable_path(rules_path),
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
        "target_score_margin": (
            ""
            if score is None
            else (
                score["black"] - score["white"]
                if target_color == "b"
                else score["white"] - score["black"]
            )
        ),
        "seed": seed,
    }


def main() -> None:
    args = parse_args()
    if args.games < 1:
        raise ValueError("--games must be at least 1.")
    if args.max_visits < 1:
        raise ValueError("--max-visits must be at least 1.")
    if args.search_depth < 1:
        raise ValueError("--search-depth must be at least 1.")
    if args.search_candidates is not None and args.search_candidates < 1:
        raise ValueError("--search-candidates must be at least 1.")
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
    results_root = SCRIPT_DIR / "results" / f"{args.board_size}x{args.board_size}_go"
    full_reports_dir = results_root / "full_reports"
    summaries_dir = results_root / "summaries"
    full_reports_dir.mkdir(parents=True, exist_ok=True)
    summaries_dir.mkdir(parents=True, exist_ok=True)

    requested_output = args.output or (
        full_reports_dir
        / f"matches-{args.search}-{args.board_size}x{args.board_size}.csv"
    )
    output = _unique_path(requested_output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if args.report_output is not None and args.report_output.exists():
        raise FileExistsError(
            f"Full report already exists and will not be overwritten: {args.report_output}"
        )
    report_path = (
        _unique_path(args.report_output)
        if args.report_output is not None
        else _unique_path(full_reports_dir / f"{output.stem}.report.json")
    )

    if args.summary_output is not None and args.summary_output.exists():
        raise FileExistsError(
            f"Summary already exists and will not be overwritten: {args.summary_output}"
        )
    summary_path = (
        _unique_path(args.summary_output)
        if args.summary_output is not None
        else _unique_path(summaries_dir / f"{output.stem}.summary.md")
    )

    output_paths = [output.resolve(), report_path.resolve(), summary_path.resolve()]
    if len(set(output_paths)) != len(output_paths):
        raise ValueError("CSV, full report and summary must use three different file paths.")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

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
    search_candidates = args.search_candidates or (5 if args.board_size == 9 else 3)
    agent = make_agent(
        agent,
        search=args.search,
        search_depth=args.search_depth,
        search_candidates=search_candidates,
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
            f"allowResignation=false,"
            f"logDir={os.path.relpath(SCRIPT_DIR / 'katago-local' / 'logs', Path.cwd())},"
            f"delayMoveScale=0,delayMoveMax=0,"
            f"searchRandSeed={args.seed},nnRandSeed={args.seed}"
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
    executable_path = (
        Path(args.katago)
        if Path(args.katago).is_file()
        else Path(shutil.which(args.katago) or args.katago)
    )
    artifacts = {
        "target_checkpoint": _file_metadata(checkpoint),
        "match_runner_source": _file_metadata(Path(__file__)),
        "negamax_source": _file_metadata(TRM_DIR / "negamax.py"),
        "agent_source": _file_metadata(TRM_DIR / "agents.py"),
        "game_environment_source": _file_metadata(TRM_DIR / "game_env.py"),
        "katago_executable": _file_metadata(executable_path),
        "katago_config": _file_metadata(args.katago_config),
        "katago_model": _file_metadata(args.katago_model),
        "katago_human_model": _file_metadata(args.katago_human_model),
        "rules": _file_metadata(args.rules),
    }
    parameters = {
        key: (
            _portable_path(value)
            if isinstance(value, Path)
            else _portable_path(value)
            if key == "katago" and Path(value).is_file()
            else _portable_argument(value)
            if isinstance(value, str)
            else value
        )
        for key, value in vars(args).items()
    }
    parameters.update(
        {
            "checkpoint": _portable_path(checkpoint),
            "requested_output": _portable_path(requested_output),
            "output": _portable_path(output),
            "report_output": _portable_path(report_path),
            "summary_output": _portable_path(summary_path),
            "komi": komi,
            "effective_gtp_rules": gtp_rules,
            "effective_max_moves": args.max_moves or 4 * args.board_size**2,
            "effective_human_profile": human_profile,
            "effective_device": device,
            "effective_search_candidates": search_candidates,
            "effective_search_depth": args.search_depth if args.search == "negamax" else 0,
        }
    )
    source_revision = subprocess.run(
        ["git", "-C", str(TRM_DIR.parent), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    source_dirty = bool(
        subprocess.run(
            ["git", "-C", str(TRM_DIR.parent), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    engine_environment, cudnn_directory = _katago_runtime_environment()
    try:
        client = GTPClient(
            command,
            timeout=args.gtp_timeout,
            environment=engine_environment,
        )
    except Exception:
        if cudnn_directory is not None:
            cudnn_directory.cleanup()
        raise
    max_moves = args.max_moves or 4 * args.board_size * args.board_size
    results: list[dict[str, Any]] = []
    started_at = datetime.now(timezone.utc)
    started_clock = time.perf_counter()
    status = "failed"
    error: str | None = None
    engine_name = "unknown"
    engine_version = "unknown"
    try:
        configure_katago(client, args.board_size, komi, gtp_rules)
        engine_name = client.command("name")
        engine_version = client.command("version")
        with output.open("x", newline="", encoding="utf-8") as result_file:
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
                results.append(result)
                writer.writerow(result)
                result_file.flush()
                print(
                    f"Game {game_number}/{args.games}: {result['target_result']} "
                    f"({result['termination']}, {result['moves']} moves)"
                )
        status = "completed"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    except KeyboardInterrupt as exc:
        error = f"{type(exc).__name__}: match interrupted by user."
        raise
    finally:
        try:
            client.close()
        finally:
            try:
                elapsed_seconds = time.perf_counter() - started_clock
                invocation = _portable_invocation([sys.executable, *sys.argv])
                try:
                    _write_report(
                        report_path,
                        status=status,
                        error=error,
                        started_at=started_at,
                        elapsed_seconds=elapsed_seconds,
                        invocation=invocation,
                        parameters=parameters,
                        engine_command=[_portable_argument(argument) for argument in command],
                        engine_name=engine_name,
                        engine_version=engine_version,
                        artifacts=artifacts,
                        results=results,
                        csv_path=output,
                        source_revision=source_revision,
                        source_dirty=source_dirty,
                    )
                finally:
                    _write_summary_report(
                        summary_path,
                        status=status,
                        error=error,
                        started_at=started_at,
                        elapsed_seconds=elapsed_seconds,
                        invocation=invocation,
                        parameters=parameters,
                        engine_name=engine_name,
                        engine_version=engine_version,
                        results=results,
                        csv_path=output,
                        report_path=report_path,
                    )
            finally:
                if cudnn_directory is not None:
                    cudnn_directory.cleanup()
        print(f"Match results: {output}")
        print(f"Match report: {report_path}")
        print(f"Match summary: {summary_path}")


if __name__ == "__main__":
    main()
