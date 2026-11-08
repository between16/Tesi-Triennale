from pathlib import Path

import numpy as np
import pytest

from model_evaluation.go_match import play_one_game as play_katago_game
from model_evaluation.go_vs_random import play_one_game as play_random_game


class _PassAgent:
    board_size = 9
    pass_logit_penalty = 0.0
    pass_penalty_before_stones = 0

    def get_move(self, env):
        return env.pass_index


class _PassingKataGo:
    def __init__(self):
        self.commands = []

    def command(self, command):
        self.commands.append(command)
        if command.startswith("genmove "):
            return "pass"
        return ""


class _ChooseLast:
    def choice(self, choices):
        return int(choices[-1])


@pytest.mark.parametrize(
    ("target_color", "expected_result"),
    [("b", "loss"), ("w", "win")],
)
def test_katago_match_assigns_outcome_for_both_balanced_colors(
    target_color, expected_result
):
    checkpoint = Path("best_model.pt")
    model = Path("katago-model.bin.gz")
    config = Path("gtp.cfg")
    rules = Path("rules.json")
    client = _PassingKataGo()

    result = play_katago_game(
        game_number=1,
        board_size=9,
        komi=7.5,
        agent=_PassAgent(),
        client=client,
        target_color=target_color,
        max_moves=10,
        seed=42,
        checkpoint=checkpoint,
        katago_model=model,
        katago_config=config,
        max_visits=1,
        rules_path=rules,
        human_model=None,
        human_profile=None,
        human_temperature_mode=None,
    )

    assert result["completed"] == "true"
    assert result["termination"] == "two_passes"
    assert result["moves"] == 2
    assert result["target_result"] == expected_result
    assert result["winner"] == "white"
    assert result["target_color"] == ("black" if target_color == "b" else "white")
    assert result["target_pass_count"] == 1
    assert result["pass_penalty_decisions"] == 0
    assert result["katago_model"] == str(model)
    assert result["katago_config"] == str(config)
    assert result["search_max_visits"] == 1
    expected_engine_move = "w" if target_color == "b" else "b"
    assert f"genmove {expected_engine_move}" in client.commands
    assert f"play {target_color} pass" in client.commands


@pytest.mark.parametrize(
    ("target_color", "expected_result"),
    [("b", "loss"), ("w", "win")],
)
def test_random_baseline_pass_control_assigns_outcome_for_both_colors(
    target_color, expected_result
):
    result = play_random_game(
        game_number=1,
        board_size=9,
        komi=7.5,
        agent=_PassAgent(),
        target_color=target_color,
        rng=_ChooseLast(),
        max_moves=10,
        checkpoint=Path("best_model.pt"),
        rules_path=Path("rules.json"),
        seed=42,
    )

    assert result["completed"] == "true"
    assert result["termination"] == "two_passes"
    assert result["moves"] == 2
    assert result["target_result"] == expected_result
    assert result["winner"] == "white"
    assert result["opponent_type"] == "random"
    assert result["opponent"] == "uniform_random_legal"
