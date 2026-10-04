import torch
import numpy as np

from models.go_trm import TinyRecursiveReasoningModel_ACTV1
from models.go_policy_head import GoPolicyHead


def apply_legal_move_mask(logits, legal_mask):
    """
    Set the logits of illegal moves to a very negative value (-1e9),
    so their probability becomes 0 after argmax or softmax.
    """
    return torch.where(legal_mask, logits, torch.tensor(-1e9, device=logits.device))


class GoTRMModel(torch.nn.Module):
    """Unified container for the TRM core and GoPolicyHead."""
    def __init__(self, config_dict):
        super(GoTRMModel, self).__init__()
        self.trm = TinyRecursiveReasoningModel_ACTV1(config_dict)
        self.policy_head = GoPolicyHead(hidden_size=config_dict["hidden_size"])

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
    """
    Agent driven by the trained TRM model.
    """
    def __init__(self, checkpoint_path="checkpoints/best_model.pt", device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load the checkpoint saved during training.
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        config_dict = checkpoint["config_dict"]

        self.model = GoTRMModel(config_dict).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def _extract_features(self, env):
        """
        Convert the current environment state into 7 input channels [1, 7, 9, 9].
        Frame 0: T (current), Frame 1: T-1, Frame 2: T-2.
        """
        tensor = np.zeros((7, env.board_size, env.board_size), dtype=np.float32)
        curr_player = env.current_player
        opp_player = 'w' if curr_player == 'b' else 'b'

        # env.history contains [board_T_minus_2, board_T_minus_1, board_T].
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

        # Channel 6: current player's color indicator (1.0 for Black, 0.0 for White).
        if curr_player == 'b':
            tensor[6, :, :] = 1.0

        return torch.from_numpy(tensor).unsqueeze(0).to(self.device)

    def get_move(self, env):
        """
        Extract features, apply the legal move mask, and return the best move index.
        """
        x = self._extract_features(env)
        legal_mask = env.get_legal_moves_mask().to(self.device)

        with torch.inference_mode():
            logits = self.model(x).squeeze(0)  # Shape: [82].

        masked_logits = apply_legal_move_mask(logits, legal_mask)
        best_move_idx = torch.argmax(masked_logits).item()
        return best_move_idx


class HumanAgent(GoAgent):
    """
    Agent for the human player in the graphical interface.
    """
    def get_move(self, env, move_idx=None):
        if move_idx is not None:
            legal_mask = env.get_legal_moves_mask()
            if legal_mask[move_idx]:
                return move_idx
            else:
                raise ValueError(f"Move {move_idx} is not legal!")
        return None  # Waiting for input from the graphical interface.