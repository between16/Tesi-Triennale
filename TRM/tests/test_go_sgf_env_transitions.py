import numpy as np
from sgfmill import sgf, sgf_moves

from game_env import GoGameEnv
from go_preprocessing import extract_features, process_sgf_file


def _make_game(board_size, moves, result):
    game = sgf.Sgf_game(size=board_size)
    game.get_root().set("RE", result)
    node = game.get_root()
    for color, coordinate in moves:
        node = node.new_child()
        node.set(color.upper(), coordinate)
    return game


def _position_key(board, board_size):
    return tuple(
        board.get(row, column) or "."
        for row in range(board_size)
        for column in range(board_size)
    )


def test_sgf_parser_preprocessing_and_environment_replay_match(tmp_path):
    board_size = 5
    moves = [
        ("b", (1, 0)),
        ("w", (1, 1)),
        ("b", (0, 1)),
        ("w", None),
        ("b", (1, 2)),
        ("w", None),
        ("b", (2, 1)),
        ("w", (4, 4)),
        ("b", None),
        ("w", None),
    ]
    sgf_path = tmp_path / "capture-and-passes.sgf"
    sgf_path.write_bytes(_make_game(board_size, moves, "B+2.5").serialise())

    parsed_game = sgf.Sgf_game.from_bytes(sgf_path.read_bytes())
    parsed_board, parsed_moves = sgf_moves.get_setup_and_moves(parsed_game)
    env = GoGameEnv(board_size=board_size, komi=2.5)
    expected_history = [None, None, parsed_board.copy()]
    expected_positions = {_position_key(parsed_board, board_size)}
    consecutive_passes = 0
    expected_features = []
    expected_targets = []

    assert len(parsed_moves) == len(moves)
    for move_number, (color, coordinate) in enumerate(parsed_moves):
        assert env.current_player == color
        assert not env.game_over
        assert _position_key(env.board, board_size) == _position_key(
            parsed_board, board_size
        )

        dataset_history = list(reversed(expected_history))
        expected_features.append(
            extract_features(dataset_history, color, board_size)
        )
        action = (
            board_size**2
            if coordinate is None
            else coordinate[0] * board_size + coordinate[1]
        )
        expected_targets.append(action)

        if coordinate is None:
            consecutive_passes += 1
        else:
            consecutive_passes = 0

        game_over = env.step(action)
        assert game_over == (consecutive_passes == 2)

        if coordinate is not None:
            row, column = coordinate
            parsed_board.play(row, column, color)
            expected_positions.add(_position_key(parsed_board, board_size))

        assert _position_key(env.board, board_size) == _position_key(
            parsed_board, board_size
        )
        assert env.consecutive_passes == consecutive_passes
        assert env.position_history == expected_positions
        assert [
            None if board is None else _position_key(board, board_size)
            for board in env.history
        ] == [
            None if board is None else _position_key(board, board_size)
            for board in expected_history[1:] + [parsed_board.copy()]
        ]
        if move_number == 6:
            assert parsed_board.get(1, 1) is None
            assert env.board.get(1, 1) is None

        expected_history = expected_history[1:] + [parsed_board.copy()]

    assert env.game_over
    assert env.consecutive_passes == 2

    features, targets, metadata = process_sgf_file(
        sgf_path,
        board_size=board_size,
        emit_warnings=False,
    )
    assert metadata["status"] == "ok"
    assert metadata["existing_passes"] == 4
    assert metadata["added_terminal_passes"] == 0
    assert targets == expected_targets
    assert len(features) == len(expected_features)
    for actual, expected in zip(features, expected_features):
        np.testing.assert_array_equal(actual, expected)
