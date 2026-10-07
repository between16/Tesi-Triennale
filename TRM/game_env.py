import numpy as np
import torch
from sgfmill import boards


class GoGameEnv:
    """Small Go environment for the TRM project.

    Rules enforced by this environment:
    - board play on an sgfmill Board;
    - PASS is always legal;
    - suicide/self-capture is forbidden;
    - positional superko is forbidden (a board position may not repeat);
    - two consecutive passes end the game;
    - final scoring uses area scoring with the configured komi.

    The environment is the single source of truth for move legality. Both the
    human GUI and the TRM agent query ``get_legal_moves_mask()`` before playing.
    """

    def __init__(self, board_size=9, komi=7.5):
        self.board_size = board_size
        self.komi = komi
        self.pass_index = board_size * board_size
        self.reset()

    # ------------------------------------------------------------------
    # State / position helpers
    # ------------------------------------------------------------------

    def reset(self):
        """Reset the board, history, turn state, and repetition history."""
        self.board = boards.Board(self.board_size)

        # Track T-2, T-1, and T for the 7-channel input.
        self.history = [None, None, self.board.copy()]

        self.current_player = "b"
        self.consecutive_passes = 0
        self.game_over = False

        # Positional superko: the board position alone identifies a position.
        # PASS does not create a new board position, so it does not need a
        # special entry here.
        self.position_history = {self._position_key(self.board)}

    def _position_key(self, board):
        """Return a hashable representation of stones on the board."""
        values = []
        for r in range(self.board_size):
            for c in range(self.board_size):
                stone = board.get(r, c)
                values.append(stone if stone is not None else ".")
        return tuple(values)

    def _test_board_move(self, r, c):
        """Try a non-pass move on a copy and return (is_legal, test_board, reason)."""
        if not (0 <= r < self.board_size and 0 <= c < self.board_size):
            return False, None, "out of bounds"

        if self.board.get(r, c) is not None:
            return False, None, "occupied intersection"

        test_board = self.board.copy()

        try:
            # sgfmill handles captures and board mutations, but it permits
            # self-capture and does not enforce ko for us.
            test_board.play(r, c, self.current_player)
        except ValueError as exc:
            return False, None, str(exc)

        # Detect self-capture. With suicide, the newly played stone/group is
        # removed again, so the played intersection is empty afterwards.
        if test_board.get(r, c) != self.current_player:
            return False, None, "suicide (self-capture)"

        # Enforce positional superko explicitly. sgfmill does not do this.
        new_position = self._position_key(test_board)
        if new_position in self.position_history:
            return False, None, "ko / repeated position"

        return True, test_board, None

    # ------------------------------------------------------------------
    # Move validation / legal moves
    # ------------------------------------------------------------------

    def is_legal_move(self, move_idx):
        """Return ``(is_legal, reason)`` without mutating the environment."""
        if self.game_over:
            return False, "game is already over"

        try:
            move_idx = int(move_idx)
        except (TypeError, ValueError):
            return False, "move index must be an integer"

        if move_idx < 0 or move_idx > self.pass_index:
            return False, "move index is out of bounds"

        if move_idx == self.pass_index:
            return True, None

        r = move_idx // self.board_size
        c = move_idx % self.board_size
        legal, _, reason = self._test_board_move(r, c)
        return legal, reason

    def get_legal_moves_mask(self):
        """Return a boolean PyTorch tensor of legal actions.

        Indices 0 .. board_size^2-1 correspond to board intersections.
        ``self.pass_index`` is PASS.
        """
        mask = torch.zeros(self.pass_index + 1, dtype=torch.bool)

        if self.game_over:
            return mask

        for r in range(self.board_size):
            for c in range(self.board_size):
                legal, _, _ = self._test_board_move(r, c)
                if legal:
                    mask[r * self.board_size + c] = True

        # PASS never changes the board and is always legal while the game is
        # active. Two passes are handled as the terminal condition in step().
        mask[self.pass_index] = True
        return mask

    # ------------------------------------------------------------------
    # State transition
    # ------------------------------------------------------------------

    def step(self, move_idx):
        """Apply a legal move and return whether the game is over.

        Illegal moves raise ValueError and leave the entire environment state
        unchanged, including board, history, current_player and pass counter.
        """
        if self.game_over:
            return True

        if not isinstance(move_idx, (int, np.integer)):
            raise ValueError(f"Move index must be an integer, got {type(move_idx).__name__}.")

        move_idx = int(move_idx)
        if move_idx < 0 or move_idx > self.pass_index:
            raise ValueError(f"Move index {move_idx} is outside 0..{self.pass_index}.")

        # PASS ----------------------------------------------------------------
        if move_idx == self.pass_index:
            self.consecutive_passes += 1

            if self.consecutive_passes >= 2:
                self.game_over = True
            else:
                self.current_player = "w" if self.current_player == "b" else "b"

            return self.game_over

        # Board move -----------------------------------------------------------
        r = move_idx // self.board_size
        c = move_idx % self.board_size

        legal, test_board, reason = self._test_board_move(r, c)
        if not legal:
            raise ValueError(f"Illegal move at ({r}, {c}): {reason}.")

        # Commit the already-validated board state. This ensures that legality
        # is checked BEFORE any state mutation or turn switch occurs.
        self.board = test_board
        self.consecutive_passes = 0

        # Update the T-2 / T-1 / T history only after the move is valid.
        self.history[0] = self.history[1]
        self.history[1] = self.history[2]
        self.history[2] = self.board.copy()

        self.position_history.add(self._position_key(self.board))

        # A normal board move cannot create the two-pass terminal condition.
        self.current_player = "w" if self.current_player == "b" else "b"
        return self.game_over

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def calculate_score(self):
        """Calculate Tromp-Taylor/area-style score for the final board."""
        black_score = 0.0
        white_score = self.komi
        visited = set()

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for r in range(self.board_size):
            for c in range(self.board_size):
                if (r, c) in visited:
                    continue

                stone = self.board.get(r, c)

                if stone == "b":
                    black_score += 1
                    visited.add((r, c))
                    continue

                if stone == "w":
                    white_score += 1
                    visited.add((r, c))
                    continue

                empty_cluster = []
                queue = [(r, c)]
                visited.add((r, c))
                surrounding_colors = set()

                while queue:
                    curr_r, curr_c = queue.pop(0)
                    empty_cluster.append((curr_r, curr_c))

                    for dr, dc in directions:
                        nr, nc = curr_r + dr, curr_c + dc
                        if not (0 <= nr < self.board_size and 0 <= nc < self.board_size):
                            continue

                        adj_stone = self.board.get(nr, nc)
                        if adj_stone is None and (nr, nc) not in visited:
                            visited.add((nr, nc))
                            queue.append((nr, nc))
                        elif adj_stone is not None:
                            surrounding_colors.add(adj_stone)

                if len(surrounding_colors) == 1:
                    color = next(iter(surrounding_colors))
                    if color == "b":
                        black_score += len(empty_cluster)
                    else:
                        white_score += len(empty_cluster)

        if black_score > white_score:
            winner = "Black"
        elif white_score > black_score:
            winner = "White"
        else:
            winner = "Draw"

        return {
            "black": black_score,
            "white": white_score,
            "winner": winner,
            "margin": abs(black_score - white_score),
        }
