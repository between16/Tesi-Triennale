from pathlib import Path

import numpy as np
import pytest
import torch
from sgfmill import boards
from torch.utils.data import DataLoader, Subset

from agents import (
    GoTRMModel,
    TRMAgent,
    apply_early_pass_penalty,
    apply_legal_move_mask,
)
from game_env import GoGameEnv
from go_dataset import GoDataset
from model_evaluation.go_eval import default_test_data_path, evaluate_model


TRM_DIR = Path(__file__).parents[1]


def _checkpoint_path(board_size):
    checkpoint_dir = TRM_DIR / "checkpoints"
    if board_size == 19:
        checkpoint_dir /= "19x19_go"
    return checkpoint_dir / "best_model.pt"


def _board_from_features(features, board_size, frame_index, player):
    player_points = []
    opponent_points = []
    player_channel = frame_index * 2
    for row in range(board_size):
        for column in range(board_size):
            if features[player_channel, row, column]:
                player_points.append((row, column))
            elif features[player_channel + 1, row, column]:
                opponent_points.append((row, column))

    board = boards.Board(board_size)
    if player == "b":
        board.apply_setup(player_points, opponent_points, [])
    else:
        board.apply_setup(opponent_points, player_points, [])
    return board


def _environment_from_features(features, board_size):
    player = "b" if bool(features[6].all()) else "w"
    assert bool(features[6].any()) == (player == "b")

    frames = [
        _board_from_features(features, board_size, frame_index, player)
        for frame_index in (2, 1, 0)
    ]
    env = GoGameEnv(board_size=board_size)
    env.history = frames
    env.board = frames[-1].copy()
    env.current_player = player
    env.position_history = {env._position_key(frame) for frame in frames}
    return env


@pytest.mark.parametrize("board_size", [9, 19])
def test_real_checkpoint_dataset_and_inference_pipeline_agree(board_size):
    checkpoint_path = _checkpoint_path(board_size)
    test_data_path = default_test_data_path(board_size)
    if not checkpoint_path.is_file() or not test_data_path.exists():
        pytest.skip(f"Local {board_size}x{board_size} checkpoint or test split is unavailable.")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config_dict"]
    assert config["board_size"] == board_size
    assert config["seq_len"] == board_size**2
    assert config["vocab_size"] == board_size**2 + 1

    dataset = GoDataset(test_data_path, augment=False, board_size=board_size)
    assert len(dataset) > 0

    model = GoTRMModel(config).cpu()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    pass_indices = np.flatnonzero(dataset.y == board_size**2)
    assert pass_indices.size > 0
    sample_indices = sorted(
        set(range(min(56, len(dataset)))) | set(pass_indices[:8].tolist())
    )
    sample_count = len(sample_indices)
    inputs = torch.stack([dataset[index][0] for index in sample_indices])
    targets = torch.stack([dataset[index][1] for index in sample_indices])
    assert tuple(inputs.shape[1:]) == (7, board_size, board_size)
    assert bool(((targets >= 0) & (targets <= board_size**2)).all())

    with torch.inference_mode():
        logits = model(inputs).float()
        direct_probabilities = torch.softmax(logits, dim=-1)

    assert tuple(logits.shape) == (sample_count, board_size**2 + 1)
    assert bool(torch.isfinite(logits).all())
    assert torch.allclose(
        direct_probabilities.sum(dim=-1),
        torch.ones(sample_count),
        atol=1e-6,
    )
    assert bool((direct_probabilities >= 0).all())

    loader = DataLoader(
        Subset(dataset, sample_indices),
        batch_size=16,
        shuffle=False,
        num_workers=0,
    )
    metrics, arrays = evaluate_model(model, loader, torch.device("cpu"), board_size)
    assert metrics["num_examples"] == sample_count
    assert metrics["pass_examples"] > 0
    assert np.array_equal(arrays["targets"], targets.numpy())
    assert np.array_equal(arrays["predictions"], logits.argmax(dim=-1).numpy())
    assert np.array_equal(
        arrays["topk_predictions"],
        logits.topk(5, dim=-1).indices.numpy(),
    )

    target_logits = logits.gather(1, targets[:, None])
    assert not bool((logits == target_logits).sum(dim=-1).gt(1).any())
    target_ranks = (logits > target_logits).sum(dim=-1) + 1
    assert metrics["top1_accuracy"] == pytest.approx(
        float((target_ranks == 1).float().mean())
    )
    assert metrics["top3_accuracy"] == pytest.approx(
        float((target_ranks <= 3).float().mean())
    )
    assert metrics["top5_accuracy"] == pytest.approx(
        float((target_ranks <= 5).float().mean())
    )
    assert metrics["mean_probability_of_correct_action"] == pytest.approx(
        float(direct_probabilities.gather(1, targets[:, None]).mean())
    )
    assert metrics["top1_accuracy"] > 0.05

    agent = TRMAgent(
        checkpoint_path=checkpoint_path,
        device="cpu",
        move_tolerance=0.0,
        rng=np.random.default_rng(42),
        pass_logit_penalty=0.0,
        pass_penalty_before_stones=0,
    )
    env = _environment_from_features(inputs[0].numpy(), board_size)
    assert tuple(agent._extract_features(env).shape) == (1, 7, board_size, board_size)
    assert torch.equal(agent._extract_features(env).cpu().squeeze(0), inputs[0])

    agent_logits, legal_mask, policy = agent.get_policy(env)
    assert torch.allclose(agent_logits.cpu(), logits[0], atol=1e-6)
    assert legal_mask.shape == policy.shape == logits[0].shape
    assert bool(legal_mask[board_size**2])
    assert torch.allclose(policy.sum(), torch.tensor(1.0), atol=1e-6)
    assert bool((policy[~legal_mask] == 0).all())

    masked_logits = apply_legal_move_mask(agent_logits, legal_mask)
    expected_policy = torch.softmax(
        apply_early_pass_penalty(masked_logits, board_size, 0, 0.0, 0),
        dim=-1,
    )
    assert torch.allclose(policy, expected_policy, atol=1e-7)
    selected_action = agent.get_move(env)
    assert selected_action == int(policy.argmax())
    assert bool(legal_mask[selected_action])
