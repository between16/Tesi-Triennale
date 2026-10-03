import numpy as np
import torch
from sgfmill import boards

class GoGameEnv:
    def __init__(self, board_size=9, komi=7.5):
        self.board_size = board_size
        self.komi = komi
        self.reset()

    def reset(self):
        """Reset the board and history for a new game."""
        self.board = boards.Board(self.board_size)
        # Track T-2, T-1, and T for the 7-channel input.
        self.history = [None, None, self.board.copy()]
        self.current_player = 'b'
        self.consecutive_passes = 0
        self.game_over = False

    def get_legal_moves_mask(self):
        """
        Generate the legal move mask [82].
        Indices 0-80: board points. Index 81: pass.
        Return a boolean PyTorch tensor (True = legal move).
        """
        mask = torch.zeros(self.board_size * self.board_size + 1, dtype=torch.bool)
        
        for r in range(self.board_size):
            for c in range(self.board_size):
                if self.board.get(r, c) is None:
                    # sgfmill checks suicide and the ko rule for us.
                    test_board = self.board.copy()
                    try:
                        test_board.play(r, c, self.current_player)
                        mask[r * self.board_size + c] = True
                    except ValueError:
                        pass  # Illegal move.
        
        mask[81] = True  # PASS is always a legal action.
        return mask

    def step(self, move_idx):
        """
        Play the move (0-80 for board points, 81 for PASS).
        Return the game_over state.
        """
        if self.game_over:
            return True

        if move_idx == 81:  # PASS
            self.consecutive_passes += 1
        else:
            r = move_idx // self.board_size
            c = move_idx % self.board_size
            try:
                self.board.play(r, c, self.current_player)
                self.consecutive_passes = 0
            except ValueError as e:
                raise ValueError(f"Illegal move attempt {r},{c}: {e}")

        # Update the history by shifting the frames backward.
        self.history[0] = self.history[1]
        self.history[1] = self.history[2]
        self.history[2] = self.board.copy()

        # Win condition: two consecutive passes.
        if self.consecutive_passes >= 2:
            self.game_over = True
        else:
            # Switch turns.
            self.current_player = 'w' if self.current_player == 'b' else 'b'

        return self.game_over

    def calculate_score(self):
        """
        Area scoring algorithm (Tromp-Taylor).
        Evaluate territories at the end of the game to determine the winner.
        """
        black_score = 0.0
        white_score = self.komi
        visited = set()

        # Helpers for traversing adjacent points (up, down, left, right).
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for r in range(self.board_size):
            for c in range(self.board_size):
                if (r, c) in visited:
                    continue

                stone = self.board.get(r, c)
                
                # If there is a stone, award 1 point directly to its owner.
                if stone == 'b':
                    black_score += 1
                    visited.add((r, c))
                elif stone == 'w':
                    white_score += 1
                    visited.add((r, c))
                else:
                    # For an empty point, use BFS to find the entire empty territory.
                    empty_cluster = []
                    queue = [(r, c)]
                    visited.add((r, c))
                    surrounding_colors = set()

                    while queue:
                        curr_r, curr_c = queue.pop(0)
                        empty_cluster.append((curr_r, curr_c))

                        for dr, dc in directions:
                            nr, nc = curr_r + dr, curr_c + dc
                            if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                                adj_stone = self.board.get(nr, nc)
                                if adj_stone is None and (nr, nc) not in visited:
                                    visited.add((nr, nc))
                                    queue.append((nr, nc))
                                elif adj_stone is not None:
                                    surrounding_colors.add(adj_stone)

                    # Assign the territory to the color that completely surrounds it.
                    if len(surrounding_colors) == 1:
                        color = list(surrounding_colors)[0]
                        if color == 'b':
                            black_score += len(empty_cluster)
                        else:
                            white_score += len(empty_cluster)

        winner = 'Black' if black_score > white_score else 'White'
        margin = abs(black_score - white_score)
        
        return {
            "black": black_score,
            "white": white_score,
            "winner": winner,
            "margin": margin
        }