"""Policy-guided Negamax search for the Go agents.

The existing TRM remains a policy network: it proposes and orders candidate
moves. Negamax searches a bounded tree of legal Go positions and evaluates
leaf positions with a lightweight heuristic. This is an inference-time
component; it does not change the model checkpoint or training pipeline.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class SearchStats:
    """Counters collected during the most recent search."""

    depth: int = 0
    nodes: int = 0
    leaves: int = 0
    cutoffs: int = 0
    policy_calls: int = 0
    score: float | None = None
    elapsed_seconds: float = 0.0


class NegamaxAgent:
    """Wrap a policy agent with depth-limited Negamax and alpha-beta pruning.

    Args:
        policy_agent: Usually a trained TRMAgent. Its get_policy(env) method
            must return (logits, legal_mask, probabilities).
        depth: Search depth in plies. A depth of 2 considers our move and the
            opponent's reply.
        max_candidates: Number of policy-ranked moves considered at each node.
            Tactical moves and PASS after an opponent PASS are added as well.
        terminal_bonus: Magnitude added to terminal wins/losses so a proven
            result takes precedence over a non-terminal heuristic estimate.
    """

    WIN_SCORE = 10_000.0

    def __init__(
        self,
        policy_agent: Any,
        depth: int = 2,
        max_candidates: int = 5,
        terminal_bonus: float = WIN_SCORE,
    ) -> None:
        if isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
            raise ValueError("depth must be a positive integer.")
        if (
            isinstance(max_candidates, bool)
            or not isinstance(max_candidates, int)
            or max_candidates < 1
        ):
            raise ValueError("max_candidates must be a positive integer.")
        if not math.isfinite(terminal_bonus) or terminal_bonus <= 0:
            raise ValueError("terminal_bonus must be finite and positive.")

        if not callable(getattr(policy_agent, "get_policy", None)):
            raise TypeError("policy_agent must expose get_policy(env).")

        self.policy_agent = policy_agent
        self.depth = depth
        self.max_candidates = max_candidates
        self.terminal_bonus = float(terminal_bonus)
        self.last_search_stats = SearchStats(depth=depth)

    def get_move(self, env) -> int:
        """Return the best move found by the bounded search."""
        if env.game_over:
            raise RuntimeError("Cannot search after the game has ended.")

        self.last_search_stats = SearchStats(depth=self.depth)
        started_at = time.perf_counter()

        color = 1 if env.current_player == "b" else -1
        score, move = self._negamax(
            env=env,
            depth=self.depth,
            alpha=float("-inf"),
            beta=float("inf"),
            color=color,
        )

        self.last_search_stats.score = score
        self.last_search_stats.elapsed_seconds = time.perf_counter() - started_at

        if move is None:
            # PASS is always legal in an active GoGameEnv, so this is a
            # defensive fallback for custom environments.
            return int(env.pass_index)
        return int(move)

    def _negamax(self, env, depth: int, alpha: float, beta: float, color: int):
        """Return (value from current player's perspective, best root move)."""
        self.last_search_stats.nodes += 1

        if env.game_over:
            self.last_search_stats.leaves += 1
            return self._terminal_value(env, color), None

        if depth <= 0:
            self.last_search_stats.leaves += 1
            return self.evaluate_position(env, color), None

        candidates, _ = self._ordered_candidates(env)

        best_value = float("-inf")
        best_move = None

        for move in candidates:
            child = self._clone_environment(env)
            child.step(move)

            child_value, _ = self._negamax(
                env=child,
                depth=depth - 1,
                alpha=-beta,
                beta=-alpha,
                color=-color,
            )
            value = -child_value

            if value > best_value:
                best_value = value
                best_move = move

            alpha = max(alpha, best_value)
            if alpha >= beta:
                self.last_search_stats.cutoffs += 1
                break

        if best_move is None:
            # The environment always exposes PASS while a game is active.
            # This branch protects against malformed custom policy masks.
            self.last_search_stats.leaves += 1
            return self.evaluate_position(env, color), int(env.pass_index)

        return best_value, best_move

    def _ordered_candidates(self, env) -> tuple[list[int], list[float]]:
        """Use TRM probabilities to order a bounded set of legal candidates."""
        import torch

        _, legal_mask, probabilities = self.policy_agent.get_policy(env)
        self.last_search_stats.policy_calls += 1

        legal = legal_mask.detach().to(device="cpu", dtype=torch.bool).tolist()
        probs = probabilities.detach().to(device="cpu", dtype=torch.float32).tolist()

        expected_actions = env.board_size * env.board_size + 1
        if len(legal) != expected_actions or len(probs) != expected_actions:
            raise ValueError(
                "Policy output size does not match the environment action space."
            )

        legal_indices = [index for index, is_legal in enumerate(legal) if is_legal]
        if not legal_indices:
            raise RuntimeError("The environment returned no legal actions.")

        pass_index = env.pass_index
        board_moves = [index for index in legal_indices if index != pass_index]
        board_moves.sort(key=lambda index: probs[index], reverse=True)

        selected = board_moves[: self.max_candidates]
        selected_set = set(selected)

        # Keep tactically forcing moves available even when the policy gives
        # them a low prior: capture an enemy group in atari or save our own.
        tactical_moves = self._tactical_moves(env, legal)
        for move in tactical_moves:
            if move not in selected_set:
                selected.append(move)
                selected_set.add(move)

        # After one PASS, the opponent must be able to end the game with PASS.
        # Include it even if it is not among the policy's highest-prior moves.
        if env.consecutive_passes == 1 and legal[pass_index]:
            if pass_index not in selected_set:
                selected.append(pass_index)
                selected_set.add(pass_index)

        # PASS is otherwise considered only when it is among the top K legal
        # actions (the policy applies its existing early-PASS penalty).
        ranked_all = sorted(legal_indices, key=lambda index: probs[index], reverse=True)
        if pass_index in ranked_all[: self.max_candidates]:
            if pass_index not in selected_set:
                selected.append(pass_index)
                selected_set.add(pass_index)

        if not selected:
            selected = [pass_index]

        # Search tactical moves first (useful for alpha-beta move ordering),
        # then preserve the policy's probability ordering.
        tactical_set = set(tactical_moves)
        selected.sort(
            key=lambda index: (
                0 if index in tactical_set else 1,
                -probs[index],
                index,
            )
        )
        return selected, probs

    @staticmethod
    def _tactical_moves(env, legal_mask: list[bool]) -> list[int]:
        """Find legal liberties of groups currently in atari."""
        board_size = env.board_size
        board = env.board
        visited: set[tuple[int, int]] = set()
        tactical: set[int] = set()
        current_player = env.current_player
        opponent = "w" if current_player == "b" else "b"

        for row in range(board_size):
            for col in range(board_size):
                color = board.get(row, col)
                if color is None or (row, col) in visited:
                    continue

                stack = [(row, col)]
                visited.add((row, col))
                liberties: set[tuple[int, int]] = set()

                while stack:
                    r, c = stack.pop()

                    for nr, nc in (
                        (r - 1, c),
                        (r + 1, c),
                        (r, c - 1),
                        (r, c + 1),
                    ):
                        if not (0 <= nr < board_size and 0 <= nc < board_size):
                            continue
                        neighbor = board.get(nr, nc)
                        if neighbor is None:
                            liberties.add((nr, nc))
                        elif neighbor == color and (nr, nc) not in visited:
                            visited.add((nr, nc))
                            stack.append((nr, nc))

                if len(liberties) == 1 and color in (current_player, opponent):
                    liberty_row, liberty_col = next(iter(liberties))
                    move = liberty_row * board_size + liberty_col
                    if legal_mask[move]:
                        tactical.add(move)

        return sorted(tactical)

    @staticmethod
    def _clone_environment(env):
        """Copy only the Go state needed for an independent search branch."""
        child = object.__new__(type(env))
        child.board_size = env.board_size
        child.komi = env.komi
        child.pass_index = env.pass_index
        child.board = env.board.copy()
        child.history = [
            board.copy() if board is not None else None
            for board in env.history
        ]
        child.current_player = env.current_player
        child.consecutive_passes = env.consecutive_passes
        child.game_over = env.game_over
        child.position_history = set(env.position_history)
        return child

    def _terminal_value(self, env, color: int) -> float:
        """Return a decisive score from the perspective of the player to move."""
        result = env.calculate_score()
        margin = float(result["black"] - result["white"])
        signed_margin = margin if color == 1 else -margin

        if signed_margin > 0:
            return self.terminal_bonus + signed_margin
        if signed_margin < 0:
            return -self.terminal_bonus + signed_margin
        return 0.0

    @staticmethod
    def evaluate_position(env, color: int) -> float:
        """Estimate the position using area score and group-liberty safety.

        The score is from Black's perspective before applying color:
        territory/area contributes one point per point of margin; groups in
        atari or with few liberties receive an additional tactical penalty.
        This is a search heuristic, not a learned Go value function.
        """
        result = env.calculate_score()
        black_minus_white = float(result["black"] - result["white"])
        value = black_minus_white

        board_size = env.board_size
        board = env.board
        visited: set[tuple[int, int]] = set()

        for row in range(board_size):
            for col in range(board_size):
                owner = board.get(row, col)
                if owner is None or (row, col) in visited:
                    continue

                stack = [(row, col)]
                visited.add((row, col))
                group_size = 0
                liberties: set[tuple[int, int]] = set()

                while stack:
                    r, c = stack.pop()
                    group_size += 1

                    for nr, nc in (
                        (r - 1, c),
                        (r + 1, c),
                        (r, c - 1),
                        (r, c + 1),
                    ):
                        if not (0 <= nr < board_size and 0 <= nc < board_size):
                            continue
                        neighbor = board.get(nr, nc)
                        if neighbor is None:
                            liberties.add((nr, nc))
                        elif neighbor == owner and (nr, nc) not in visited:
                            visited.add((nr, nc))
                            stack.append((nr, nc))

                liberty_count = len(liberties)
                if liberty_count == 1:
                    risk = 2.5 + 1.25 * group_size
                elif liberty_count == 2:
                    risk = 0.5 + 0.30 * group_size
                elif liberty_count == 3:
                    risk = 0.08 * group_size
                else:
                    risk = 0.0

                # A vulnerable Black group lowers Black's score; a vulnerable
                # White group raises Black's relative score, and vice versa.
                value += -risk if owner == "b" else risk

        return value * color
