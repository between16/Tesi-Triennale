import torch

from models.go_policy_head import GoPolicyHead
from models.go_trm import TinyRecursiveReasoningModel_ACTV1


class GoTRMModel(torch.nn.Module):
    """Unified container for the TRM core and Go policy head."""

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
