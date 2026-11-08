import csv
from pathlib import Path
import sys

import numpy as np
import pytest

from model_evaluation.go_elo import fit_elo, load_rank_thresholds, rank_for_rating
from model_evaluation.go_match import (
    GTPClient,
    action_to_gtp,
    gtp_to_action,
    load_board_rules,
    parse_args,
)
from model_evaluation.go_vs_random import random_legal_action
from game_env import GoGameEnv
from model_evaluation.go_pass_calibrate import pass_decision_metrics


@pytest.mark.parametrize("board_size", [9, 19])
def test_evaluation_rules_match_custom_environment(board_size):
    rules_path = (
        Path(__file__).parents[1] / "model_evaluation" / "config" / "rules.json"
    )
    komi, gtp_rules = load_board_rules(rules_path, board_size)

    assert komi == 7.5
    assert gtp_rules == {
        "ko": "POSITIONAL",
        "scoring": "AREA",
        "suicide": False,
        "tax": "NONE",
        "hasButton": False,
        "whiteHandicapBonus": "0",
        "friendlyPassOk": True,
    }


@pytest.mark.parametrize("board_size", [9, 19])
def test_gtp_coordinates_round_trip(board_size):
    for action in (0, board_size - 1, board_size * (board_size - 1), board_size**2 - 1):
        assert gtp_to_action(action_to_gtp(action, board_size), board_size) == action
    assert action_to_gtp(board_size**2, board_size) == "pass"
    assert gtp_to_action("pass", board_size) == board_size**2


def test_gtp_coordinates_skip_letter_i_and_reject_invalid_values():
    assert action_to_gtp(8, 9) == "J9"
    assert gtp_to_action("J9", 9) == 8
    with pytest.raises(ValueError, match="outside"):
        gtp_to_action("T19", 9)


def test_human_temperature_mode_is_exposed_by_match_runner(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "go_match.py",
            "--board-size",
            "9",
            "--katago-config",
            "gtp.cfg",
            "--katago-model",
            "model.bin",
            "--human-temperature-mode",
            "full",
        ],
    )

    assert parse_args().human_temperature_mode == "full"


@pytest.mark.parametrize("board_size", [9, 19])
def test_random_baseline_only_selects_legal_moves(board_size):
    env = GoGameEnv(board_size=board_size)
    rng = np.random.default_rng(42)

    for _ in range(10):
        action = random_legal_action(env, rng)
        assert env.is_legal_move(action) == (True, None)
        env.step(action)


def test_pass_calibration_metrics_reflect_early_penalty():
    report = pass_decision_metrics(
        pass_margins=np.array([3.0, 2.0, -1.0, -2.0]),
        occupied_counts=np.array([2, 20, 2, 20]),
        targets=np.array([0, 4, 4, 0]),
        pass_action=4,
        penalty=3.5,
        before_stones=8,
    )

    assert report["pass_precision"] == pytest.approx(1.0)
    assert report["pass_recall"] == pytest.approx(0.5)
    assert report["false_passes"] == 0
    assert report["missed_passes"] == 1


def test_gtp_client_reads_responses_and_surfaces_engine_errors():
    fake_engine = (
        "import sys\n"
        "for line in sys.stdin:\n"
        " command = line.strip()\n"
        " status = '?' if command == 'bad' else '='\n"
        " print(status + ' response', flush=True)\n"
        " print(flush=True)\n"
    )
    client = GTPClient([sys.executable, "-u", "-c", fake_engine], timeout=2)
    try:
        assert client.command("protocol_version") == "response"
        with pytest.raises(RuntimeError, match="rejected"):
            client.command("bad")
    finally:
        client.close()


def test_elo_fit_uses_fractional_draws_and_filters_incomplete_games():
    matches = [
        {"completed": "true", "board_size": "9", "target_result": "win"},
        {"completed": "true", "board_size": "9", "target_result": "draw"},
        {"completed": "true", "board_size": "9", "target_result": "loss"},
        {"completed": "false", "board_size": "9", "target_result": "incomplete"},
        {"completed": "true", "board_size": "19", "target_result": "win"},
    ]

    report = fit_elo(matches, board_size=9)

    assert report["games"] == 3
    assert (report["wins"], report["draws"], report["losses"]) == (1, 1, 1)
    assert report["elo_difference"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("outcomes", "status"),
    [(["win", "win"], "unbounded_above"), (["loss", "loss"], "unbounded_below")],
)
def test_elo_fit_reports_unbounded_boundary_mles(outcomes, status):
    report = fit_elo(
        [{"completed": "true", "board_size": "9", "target_result": result} for result in outcomes]
    )

    assert report["rating_status"] == status
    assert report["elo_difference"] is None


def test_elo_fit_rejects_mixed_board_sizes_and_reference_configurations():
    base = {
        "completed": "true",
        "board_size": "9",
        "target_result": "win",
        "katago_model": "reference-a.bin",
        "katago_config": "gtp.cfg",
        "search_max_visits": "16",
        "pass_logit_penalty": "0",
        "pass_penalty_before_stones": "0",
        "human_temperature_mode": "full",
    }
    with pytest.raises(ValueError, match="multiple board sizes"):
        fit_elo(
            [
                base,
                {**base, "board_size": "19"},
            ]
        )
    with pytest.raises(ValueError, match="multiple katago_model"):
        fit_elo(
            [
                base,
                {**base, "target_result": "loss", "katago_model": "reference-b.bin"},
            ],
            board_size=9,
        )
    with pytest.raises(ValueError, match="multiple search_max_visits"):
        fit_elo(
            [
                base,
                {**base, "target_result": "loss", "search_max_visits": "64"},
            ],
            board_size=9,
        )
    with pytest.raises(ValueError, match="multiple pass_logit_penalty"):
        fit_elo(
            [
                base,
                {**base, "target_result": "loss", "pass_logit_penalty": "3.5"},
            ],
            board_size=9,
        )
    with pytest.raises(ValueError, match="multiple human_temperature_mode"):
        fit_elo(
            [
                base,
                {**base, "target_result": "loss", "human_temperature_mode": "engine_default"},
            ],
            board_size=9,
        )


def test_rank_calibration_selects_highest_eligible_threshold(tmp_path):
    calibration = tmp_path / "ranks.csv"
    with calibration.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["rank", "elo_min"])
        writer.writeheader()
        writer.writerows(
            [
                {"rank": "20k", "elo_min": 1000},
                {"rank": "1d", "elo_min": 2000},
                {"rank": "3d", "elo_min": 2200},
            ]
        )

    thresholds = load_rank_thresholds(calibration)

    assert rank_for_rating(2150, thresholds) == "1d"
    assert rank_for_rating(900, thresholds) == "below 20k"
