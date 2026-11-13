import pytest
import torch
from sgfmill import boards

from game_env import GoGameEnv
from negamax import NegamaxAgent


class UniformPolicy:
    """Small deterministic policy stub used to test the search independently."""

    def __init__(self, preferred_move=None):
        self.preferred_move = preferred_move
        self.calls = 0

    def get_policy(self, env):
        self.calls += 1
        legal_mask = env.get_legal_moves_mask()
        probabilities = legal_mask.to(dtype=torch.float32)
        probabilities /= probabilities.sum()

        if self.preferred_move is not None:
            move = self.preferred_move
            if move < probabilities.numel() and bool(legal_mask[move]):
                probabilities.zero_()
                probabilities[legal_mask] = 0.001
                probabilities[move] = 0.976
                probabilities /= probabilities.sum()

        logits = torch.log(probabilities.clamp_min(1e-12))
        return logits, legal_mask, probabilities


def _set_board(env, stones):
    env.board = boards.Board(env.board_size)
    for color, row, col in stones:
        env.board.play(row, col, color)
    env.history = [None, None, env.board.copy()]
    env.position_history = {env._position_key(env.board)}
    env.current_player = "b"
    env.consecutive_passes = 0
    env.game_over = False


def test_negamax_returns_legal_move_without_mutating_root_state():
    env = GoGameEnv(board_size=5)
    before_position = env._position_key(env.board)
    before_player = env.current_player
    policy = UniformPolicy()
    agent = NegamaxAgent(policy, depth=2, max_candidates=2)

    move = agent.get_move(env)

    assert isinstance(move, int)
    assert bool(env.get_legal_moves_mask()[move])
    assert env._position_key(env.board) == before_position
    assert env.current_player == before_player
    assert env.consecutive_passes == 0
    assert agent.last_search_stats.nodes > 1
    assert agent.last_search_stats.policy_calls == policy.calls
    assert agent.last_search_stats.leaves > 0


def test_tactical_atari_liberty_is_included_even_outside_policy_top_k():
    env = GoGameEnv(board_size=5)
    _set_board(
        env,
        [
            ("b", 2, 2),
            ("w", 1, 2),
            ("w", 2, 1),
            ("w", 3, 2),
        ],
    )
    saving_move = 2 * env.board_size + 3
    policy = UniformPolicy(preferred_move=0)
    agent = NegamaxAgent(policy, depth=1, max_candidates=1)

    candidates, _ = agent._ordered_candidates(env)

    assert saving_move in candidates
    assert bool(env.get_legal_moves_mask()[saving_move])


def test_pass_is_available_after_opponent_has_passed():
    env = GoGameEnv(board_size=5)
    env.step(env.pass_index)
    assert env.consecutive_passes == 1

    policy = UniformPolicy(preferred_move=0)
    agent = NegamaxAgent(policy, depth=1, max_candidates=1)

    candidates, _ = agent._ordered_candidates(env)

    assert env.pass_index in candidates


def test_clone_environment_is_independent():
    env = GoGameEnv(board_size=5)
    child = NegamaxAgent._clone_environment(env)
    move = 2 * env.board_size + 2

    child.step(move)

    assert env.board.get(2, 2) is None
    assert child.board.get(2, 2) == "b"
    assert env.current_player == "b"
    assert child.current_player == "w"
    assert env.position_history != child.position_history


def test_terminal_value_uses_correct_player_perspective_after_two_passes():
    env = GoGameEnv(board_size=5, komi=7.5)
    env.step(env.pass_index)
    env.step(env.pass_index)
    assert env.game_over

    agent = NegamaxAgent(UniformPolicy(), depth=1, max_candidates=1)

    # An empty board with 7.5 komi is a White win.
    assert agent._terminal_value(env, color=1) < 0
    assert agent._terminal_value(env, color=-1) > 0


def test_evaluation_changes_sign_with_perspective():
    env = GoGameEnv(board_size=5)
    env.step(12)  # Black takes the centre.
    agent = NegamaxAgent(UniformPolicy(), depth=1, max_candidates=1)

    black_value = agent.evaluate_position(env, color=1)
    white_value = agent.evaluate_position(env, color=-1)

    assert black_value == pytest.approx(-white_value)


def test_search_rejects_terminal_environment():
    env = GoGameEnv(board_size=5)
    env.step(env.pass_index)
    env.step(env.pass_index)
    agent = NegamaxAgent(UniformPolicy(), depth=1, max_candidates=1)

    with pytest.raises(RuntimeError, match="game has ended"):
        agent.get_move(env)
