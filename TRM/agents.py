import math
from pathlib import Path

import numpy as np
import torch

from models.go_trm import TinyRecursiveReasoningModel_ACTV1
from models.go_policy_head import GoPolicyHead

DEFAULT_PASS_LOGIT_PENALTY = 3.5
DEFAULT_PASS_PENALTY_BEFORE_STONES = 16


def apply_legal_move_mask(
    logits: torch.Tensor,
    legal_mask: torch.Tensor,
) -> torch.Tensor:
    """Mask illegal actions so their softmax probability is exactly zero."""
    if logits.ndim != 1:
        raise ValueError(
            f"Logits must be one-dimensional, received shape {tuple(logits.shape)}."
        )

    legal_mask = torch.as_tensor(
        legal_mask,
        device=logits.device,
        dtype=torch.bool,
    )

    if legal_mask.shape != logits.shape:
        raise ValueError(
            "Logits and legal mask must have the same shape; "
            f"received {tuple(logits.shape)} and {tuple(legal_mask.shape)}."
        )

    if not bool(legal_mask.any().item()):
        raise ValueError("No legal actions available.")

    if not bool(torch.isfinite(logits[legal_mask]).all().item()):
        raise ValueError("Non-finite logits found for legal actions.")

    return logits.masked_fill(~legal_mask, float("-inf"))


def apply_early_pass_penalty(
    logits: torch.Tensor,
    board_size: int,
    occupied_stones: int,
    penalty: float,
    before_stones: int,
) -> torch.Tensor:
    """Lower the PASS logit only while fewer than ``before_stones`` are occupied."""
    if logits.ndim != 1 or logits.numel() != board_size * board_size + 1:
        raise ValueError("Logits must contain one action per intersection plus PASS.")
    if (
        isinstance(occupied_stones, bool)
        or not isinstance(occupied_stones, int)
        or occupied_stones < 0
    ):
        raise ValueError("occupied_stones must be a non-negative integer.")
    if not math.isfinite(penalty) or penalty < 0:
        raise ValueError("penalty must be finite and non-negative.")
    if (
        isinstance(before_stones, bool)
        or not isinstance(before_stones, int)
        or before_stones < 0
    ):
        raise ValueError("before_stones must be a non-negative integer.")
    if (penalty == 0) != (before_stones == 0):
        raise ValueError(
            "Set both penalty and before_stones to zero to disable the adjustment, "
            "or set both to positive values."
        )
    if penalty == 0 or occupied_stones >= before_stones:
        return logits
    adjusted_logits = logits.clone()
    adjusted_logits[-1] -= penalty
    return adjusted_logits


def select_near_best_action(
    probabilities: torch.Tensor | np.ndarray,
    tolerance: float = 0.02,
    rng: np.random.Generator | None = None,
    legal_mask: torch.Tensor | np.ndarray | None = None,
) -> int:
    """Choose randomly among legal actions close to the best probability.

    ``tolerance`` is relative to the best legal probability. For example,
    ``tolerance=0.02`` keeps actions whose probability is at least 98% of the
    best legal probability. This avoids admitting zero-probability illegal
    actions when the best probability is small, as can happen on a 19x19 board.

    Pass ``legal_mask`` whenever it is available. If it is omitted, actions
    with zero probability are treated as unavailable.
    """
    if not np.isfinite(tolerance) or not 0.0 <= tolerance < 1.0:
        raise ValueError("tolerance must be non-negative and smaller than 1.")

    if isinstance(probabilities, torch.Tensor):
        action_probabilities = probabilities.detach().cpu().numpy()
    else:
        action_probabilities = np.asarray(probabilities)

    action_probabilities = np.asarray(action_probabilities, dtype=np.float64)

    if action_probabilities.ndim != 1 or action_probabilities.size == 0:
        raise ValueError(
            "probabilities must be a non-empty one-dimensional array."
        )

    if (
        not np.isfinite(action_probabilities).all()
        or (action_probabilities < 0.0).any()
    ):
        raise ValueError("probabilities must be finite and non-negative.")

    if legal_mask is None:
        valid_mask = action_probabilities > 0.0
    elif isinstance(legal_mask, torch.Tensor):
        valid_mask = legal_mask.detach().cpu().numpy().astype(bool)
    else:
        valid_mask = np.asarray(legal_mask, dtype=bool)

    if valid_mask.shape != action_probabilities.shape:
        raise ValueError(
            "legal_mask and probabilities must have the same shape."
        )

    if not valid_mask.any():
        raise ValueError("No legal actions available.")

    best_probability = float(action_probabilities[valid_mask].max())
    if best_probability <= 0.0:
        raise ValueError("No legal action has positive probability.")

    # Use a relative threshold, not an absolute subtraction such as 0.02.
    threshold = best_probability * (1.0 - tolerance)
    candidates = np.flatnonzero(
        valid_mask & (action_probabilities >= threshold)
    )

    if candidates.size == 0:
        # This should not happen because the maximum legal action is a candidate.
        raise RuntimeError("Near-best action selection produced no candidates.")

    if rng is None:
        rng = np.random.default_rng()

    return int(rng.choice(candidates))


class GoTRMModel(torch.nn.Module):
    """Unified container for the TRM core and GoPolicyHead."""

    def __init__(self, config_dict):
        super().__init__()
        self.trm = TinyRecursiveReasoningModel_ACTV1(config_dict)
        self.policy_head = GoPolicyHead(
            hidden_size=config_dict["hidden_size"],
            board_size=config_dict.get("board_size", 9),
        )

    def forward(self, x):
        batch_dict = {"inputs": x}
        carry = self.trm.initial_carry(batch_dict)
        carry, hidden_features = self.trm(carry, batch_dict)
        return self.policy_head(hidden_features)


class GoAgent:
    """Abstract base class for all Go agents."""

    def get_move(self, env):
        raise NotImplementedError


class TRMAgent(GoAgent):
    """Agent driven by the trained TRM model."""

    def __init__(
        self,
        checkpoint_path: str | Path = "checkpoints/best_model.pt",
        device=None,
        move_tolerance=0.02,
        rng=None,
        pass_logit_penalty=DEFAULT_PASS_LOGIT_PENALTY,
        pass_penalty_before_stones=DEFAULT_PASS_PENALTY_BEFORE_STONES,
    ) -> None:
        if not 0.0 <= move_tolerance < 1.0:
            raise ValueError("move_tolerance must be between 0 and 1.")

        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        config_dict = checkpoint["config_dict"]

        self.model = GoTRMModel(config_dict).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.board_size = config_dict.get("board_size", 9)
        self.move_tolerance = float(move_tolerance)
        if not math.isfinite(pass_logit_penalty) or pass_logit_penalty < 0:
            raise ValueError("pass_logit_penalty must be finite and non-negative.")
        if (
            isinstance(pass_penalty_before_stones, bool)
            or not isinstance(pass_penalty_before_stones, int)
            or pass_penalty_before_stones < 0
        ):
            raise ValueError("pass_penalty_before_stones must be a non-negative integer.")
        if (pass_logit_penalty == 0) != (pass_penalty_before_stones == 0):
            raise ValueError(
                "Set both PASS penalty options to zero to disable the adjustment, "
                "or set both to positive values."
            )
        self.pass_logit_penalty = float(pass_logit_penalty)
        self.pass_penalty_before_stones = pass_penalty_before_stones
        self.rng = rng if rng is not None else np.random.default_rng()

    def _extract_features(self, env):
        """Convert the current Go state into [1, 7, board_size, board_size]."""
        tensor = np.zeros(
            (7, env.board_size, env.board_size),
            dtype=np.float32,
        )

        curr_player = env.current_player
        opp_player = "w" if curr_player == "b" else "b"

        # env.history contains [T-2, T-1, T].
        frames = [env.history[2], env.history[1], env.history[0]]

        for frame_idx, board_state in enumerate(frames):
            if board_state is None:
                continue

            p_ch = frame_idx * 2
            o_ch = frame_idx * 2 + 1

            for r in range(env.board_size):
                for c in range(env.board_size):
                    stone = board_state.get(r, c)
                    if stone == curr_player:
                        tensor[p_ch, r, c] = 1.0
                    elif stone == opp_player:
                        tensor[o_ch, r, c] = 1.0

        # Channel 6: current player's color indicator.
        if curr_player == "b":
            tensor[6, :, :] = 1.0

        return torch.from_numpy(tensor).unsqueeze(0).to(self.device)

    def get_policy(self, env):
        """Return raw logits, legal mask, and masked policy probabilities."""
        if env.game_over:
            raise RuntimeError("Cannot calculate a move after the game has ended.")

        if env.board_size != self.board_size:
            raise ValueError(
                f"Agent checkpoint expects a {self.board_size}x{self.board_size} "
                f"board, but environment uses {env.board_size}x{env.board_size}."
            )

        x = self._extract_features(env)
        legal_mask = env.get_legal_moves_mask().to(self.device)

        with torch.inference_mode():
            logits = self.model(x).squeeze(0)

        expected_actions = self.board_size * self.board_size + 1
        if logits.shape != (expected_actions,):
            raise RuntimeError(
                f"Expected {expected_actions} logits, "
                f"received shape {tuple(logits.shape)}."
            )

        if legal_mask.shape != logits.shape:
            raise RuntimeError(
                f"Logits shape {tuple(logits.shape)} does not match "
                f"legal mask shape {tuple(legal_mask.shape)}."
            )

        masked_logits = apply_legal_move_mask(logits, legal_mask)
        if self.pass_logit_penalty:
            occupied_stones = sum(
                env.board.get(row, column) is not None
                for row in range(self.board_size)
                for column in range(self.board_size)
            )
            masked_logits = apply_early_pass_penalty(
                masked_logits,
                self.board_size,
                occupied_stones,
                self.pass_logit_penalty,
                self.pass_penalty_before_stones,
            )
        probabilities = torch.softmax(masked_logits, dim=-1)
        return logits, legal_mask, probabilities

    def get_move(self, env):
        """Return a near-best legal action, randomizing only among close moves."""
        _, legal_mask, probabilities = self.get_policy(env)
        return select_near_best_action(
            probabilities,
            tolerance=self.move_tolerance,
            rng=self.rng,
            legal_mask=legal_mask,
        )


class HumanAgent(GoAgent):
    """Agent used by the graphical interface for a human player."""

    def get_move(self, env, move_idx=None):
        if move_idx is None:
            return None

        legal_mask = env.get_legal_moves_mask()
        if not isinstance(move_idx, (int, np.integer)):
            raise ValueError("Move index must be an integer.")

        if move_idx < 0 or move_idx >= legal_mask.numel():
            raise ValueError(f"Move index {move_idx} is out of bounds.")

        if bool(legal_mask[move_idx].item()):
            return int(move_idx)

        raise ValueError(f"Move {move_idx} is not legal!")
