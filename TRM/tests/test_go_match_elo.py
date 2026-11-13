import csv
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import pytest

from model_evaluation.go_elo import fit_elo, load_rank_thresholds, rank_for_rating
from model_evaluation.go_match import (
    GTPClient,
    _comparison_key,
    _match_summary,
    _portable_argument,
    _portable_invocation,
    _portable_path,
    _unique_path,
    action_to_gtp,
    gtp_to_action,
    load_board_rules,
    make_agent,
    parse_args,
    _write_report,
)
from negamax import NegamaxAgent
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


def test_negamax_options_are_exposed_by_match_runner(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "go_match.py",
            "--board-size",
            "19",
            "--katago-config",
            "gtp.cfg",
            "--katago-model",
            "model.bin",
            "--search",
            "negamax",
            "--search-depth",
            "2",
            "--search-candidates",
            "3",
        ],
    )

    args = parse_args()

    assert args.search == "negamax"
    assert args.search_depth == 2
    assert args.search_candidates == 3


def test_make_agent_wraps_policy_with_negamax():
    class PolicyStub:
        def get_policy(self, env):
            raise NotImplementedError

    policy = PolicyStub()

    agent = make_agent(policy, "negamax", search_depth=2, search_candidates=3)

    assert isinstance(agent, NegamaxAgent)
    assert agent.policy_agent is policy
    assert agent.depth == 2
    assert agent.max_candidates == 3
    assert make_agent(policy, "policy", 2, 3) is policy


def test_match_summary_reports_color_balance_and_excludes_incomplete_results():
    base = {
        "completed": "true",
        "termination": "two_passes",
        "moves": 80,
        "target_pass_count": 1,
        "target_passes_before_16_stones": 0,
        "pass_penalty_decisions": 16,
        "target_score_margin": 4.5,
        "search_nodes": 0,
        "search_leaves": 0,
        "search_cutoffs": 0,
        "search_policy_calls": 0,
        "search_time_seconds": 0.0,
    }
    report = _match_summary(
        [
            {**base, "target_result": "win", "target_color": "black"},
            {**base, "target_result": "loss", "target_color": "white"},
            {
                **base,
                "completed": "false",
                "termination": "move_limit",
                "target_result": "incomplete",
                "target_color": "black",
                "target_score_margin": "",
            },
        ]
    )

    assert report["requested_games"] == 3
    assert report["games_recorded"] == 3
    assert report["not_started_games"] == 0
    assert report["completed_games"] == 2
    assert report["incomplete_games"] == 1
    assert (report["wins"], report["draws"], report["losses"]) == (1, 0, 1)
    assert report["score_rate"] == pytest.approx(0.5)
    assert report["target_color_results"]["black"]["wins"] == 1
    assert report["target_color_results"]["white"]["losses"] == 1
    assert report["termination_counts"] == {"move_limit": 1, "two_passes": 2}


def test_output_path_preserves_existing_match_files(tmp_path):
    requested = tmp_path / "match.csv"
    requested.write_text("existing", encoding="utf-8")

    unique = _unique_path(requested)

    assert unique != requested
    assert unique.name.startswith("match.run-")
    assert requested.read_text(encoding="utf-8") == "existing"


def test_report_paths_are_repository_relative_not_machine_absolute():
    repo_root = Path(__file__).parents[2]
    internal_path = repo_root / "TRM" / "checkpoints" / "best_model.pt"

    assert _portable_path(internal_path) == "TRM/checkpoints/best_model.pt"
    assert _portable_argument(str(internal_path)) == "TRM/checkpoints/best_model.pt"
    assert _portable_argument("logDir=" + str(repo_root / "TRM" / "logs")) == "logDir=TRM/logs"
    invocation = _portable_invocation(
        ["/some/python/bin/python", str(repo_root / "TRM" / "app.py"), "--board-size", "9"]
    )
    assert invocation.startswith("python3 TRM/app.py")
    assert str(repo_root) not in invocation


def test_json_report_records_parameters_and_csv_hash(tmp_path):
    results_csv = tmp_path / "run.csv"
    results_csv.write_text("game,target_result\n1,win\n", encoding="utf-8")
    report_path = tmp_path / "run.report.json"

    _write_report(
        report_path,
        status="completed",
        error=None,
        started_at=datetime.now(timezone.utc),
        elapsed_seconds=2.5,
        invocation="python3 model_evaluation/go_match.py --search negamax",
        parameters={"search": "negamax", "search_depth": 2},
        engine_command=["katago", "gtp"],
        engine_name="KataGo",
        engine_version="1.18.2",
        artifacts={},
        results=[],
        csv_path=results_csv,
        source_revision="revision",
        source_dirty=False,
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "completed"
    assert report["parameters"]["search"] == "negamax"
    assert report["engine"]["version"] == "1.18.2"
    assert report["results_csv"]["sha256"]
    assert report["summary"]["completed_games"] == 0


def test_comparison_key_ignores_search_configuration():
    common = {
        "board_size": 9,
        "effective_search_depth": 0,
        "effective_search_candidates": 5,
    }
    artifacts = {"model": {"sha256": "abc"}}

    policy_key = _comparison_key(
        {**common, "search": "policy"},
        artifacts,
    )
    negamax_key = _comparison_key(
        {
            **common,
            "search": "negamax",
            "effective_search_depth": 2,
            "effective_search_candidates": 3,
        },
        artifacts,
    )

    assert policy_key == negamax_key


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
    with pytest.raises(ValueError, match="multiple search_mode"):
        fit_elo(
            [
                {**base, "search_mode": "policy"},
                {**base, "target_result": "loss", "search_mode": "negamax"},
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
