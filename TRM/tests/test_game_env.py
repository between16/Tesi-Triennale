import pytest
from sgfmill import boards

from game_env import GoGameEnv


def _set_board(env, stones):
    """Place setup stones directly, then rebuild repetition history."""
    env.board = boards.Board(env.board_size)
    for colour, row, col in stones:
        env.board.play(row, col, colour)

    env.history = [None, None, env.board.copy()]
    env.position_history = {env._position_key(env.board)}
    env.current_player = "b"
    env.consecutive_passes = 0
    env.game_over = False


def test_suicide_is_illegal_and_state_is_unchanged():
    env = GoGameEnv(board_size=5)

    # White stones at the four orthogonal neighbours of the centre.
    _set_board(
        env,
        [
            ("w", 1, 2),
            ("w", 2, 1),
            ("w", 2, 3),
            ("w", 3, 2),
        ],
    )

    center = 2 * env.board_size + 2
    legal, reason = env.is_legal_move(center)
    assert not legal
    assert reason == "suicide (self-capture)"

    before = env._position_key(env.board)
    before_player = env.current_player

    with pytest.raises(ValueError, match="suicide"):
        env.step(center)

    assert env._position_key(env.board) == before
    assert env.current_player == before_player
    assert env.consecutive_passes == 0


def test_pass_is_legal_and_two_passes_end_game():
    env = GoGameEnv(board_size=5)

    assert bool(env.get_legal_moves_mask()[env.pass_index])
    assert env.step(env.pass_index) is False
    assert env.current_player == "w"
    assert env.consecutive_passes == 1

    assert env.step(env.pass_index) is True
    assert env.game_over is True
    assert env.consecutive_passes == 2


def test_occupied_point_is_illegal():
    env = GoGameEnv(board_size=5)
    env.board.play(0, 0, "b")
    env.position_history = {env._position_key(env.board)}

    legal, reason = env.is_legal_move(0)
    assert not legal
    assert reason == "occupied intersection"


def test_superko_rejects_a_repeated_position():
    env = GoGameEnv(board_size=5)
    move = 2 * env.board_size + 2

    # Add the position that would result from playing at the centre to the
    # repetition history. This isolates the superko check from the capture
    # mechanics and verifies that repeated board states are rejected.
    candidate = env.board.copy()
    candidate.play(2, 2, "b")
    env.position_history.add(env._position_key(candidate))

    legal, reason = env.is_legal_move(move)
    assert not legal
    assert reason == "ko / repeated position"
