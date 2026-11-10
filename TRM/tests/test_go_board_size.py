import inspect

import numpy as np
import pytest
import torch

from go_dataset import apply_board_symmetry
from models.go_model import GoTRMModel as SharedGoTRMModel
from app import checkpoint_for_board_size
from train import GoTRMModel, evaluate
from agents import (
    DEFAULT_PASS_LOGIT_PENALTY,
    DEFAULT_PASS_PENALTY_BEFORE_STONES,
    TRMAgent,
    apply_early_pass_penalty,
    apply_legal_move_mask,
    select_near_best_action,
)


def test_training_and_inference_import_the_shared_model():
    from agents import GoTRMModel as AgentGoTRMModel
    from micro_train import GoTRMModel as MicroTrainGoTRMModel
    from model_evaluation.go_eval import GoTRMModel as EvaluatorGoTRMModel

    assert GoTRMModel is SharedGoTRMModel
    assert AgentGoTRMModel is SharedGoTRMModel
    assert MicroTrainGoTRMModel is SharedGoTRMModel
    assert EvaluatorGoTRMModel is SharedGoTRMModel


@pytest.mark.parametrize(
    ("board_size", "checkpoint_name"),
    [(9, "best_model.pt"), (19, "19x19_go/best_model.pt")],
)
def test_gui_selects_checkpoint_for_board_size(board_size, checkpoint_name):
    checkpoint_path = checkpoint_for_board_size(board_size)

    assert checkpoint_path.as_posix().endswith(checkpoint_name)


@pytest.mark.parametrize("board_size", [9, 19])
def test_model_outputs_one_logit_per_intersection_and_pass(board_size):
    board_area = board_size * board_size
    config = {
        "batch_size": 1,
        "board_size": board_size,
        "seq_len": board_area,
        "in_channels": 7,
        "vocab_size": board_area + 1,
        "H_cycles": 1,
        "L_cycles": 1,
        "H_layers": 1,
        "L_layers": 1,
        "hidden_size": 16,
        "expansion": 2.0,
        "num_heads": 4,
        "pos_encodings": "none",
        "forward_dtype": "float32",
        "mlp_t": False,
    }

    model = GoTRMModel(config)
    logits = model(torch.randn(1, 7, board_size, board_size))

    assert logits.shape == (1, board_area + 1)


@pytest.mark.parametrize("board_size", [9, 19])
def test_board_symmetry_preserves_pass_and_board_shape(board_size):
    board = np.zeros((7, board_size, board_size), dtype=np.float32)
    move = board_size * (board_size - 1) + 1

    transformed_board, transformed_move = apply_board_symmetry(
        board, move, symmetry_idx=1
    )
    transformed_pass_board, transformed_pass = apply_board_symmetry(
        board, board_size * board_size, symmetry_idx=7
    )

    assert transformed_board.shape == board.shape
    assert 0 <= transformed_move < board_size * board_size
    assert transformed_pass_board.shape == board.shape
    assert transformed_pass == board_size * board_size


def test_near_best_selection_randomizes_only_within_relative_tolerance():
    probabilities = torch.tensor([0.51, 0.50, 0.48, 0.0])
    rng = np.random.default_rng(42)

    selected = {
        select_near_best_action(probabilities, tolerance=0.02, rng=rng)
        for _ in range(100)
    }

    # 0.50 is within 2% of 0.51; 0.48 is not.
    assert selected == {0, 1}


def test_near_best_selection_keeps_clear_best_action():
    probabilities = torch.tensor([0.80, 0.77, 0.0])

    assert select_near_best_action(
        probabilities, tolerance=0.02, rng=np.random.default_rng(42)
    ) == 0


def test_near_best_selection_excludes_zero_probability_actions_on_19x19():
    probabilities = torch.zeros(362)
    probabilities[10] = 0.009
    probabilities[11] = 0.00895

    legal_mask = torch.zeros(362, dtype=torch.bool)
    legal_mask[[10, 11, 361]] = True

    rng = np.random.default_rng(42)
    selected = {
        select_near_best_action(
            probabilities,
            tolerance=0.02,
            rng=rng,
            legal_mask=legal_mask,
        )
        for _ in range(100)
    }

    assert selected == {10, 11}
    assert all(bool(legal_mask[action]) for action in selected)


def test_near_best_selection_respects_explicit_legal_mask():
    probabilities = torch.tensor([0.50, 0.499, 0.498, 0.0])
    legal_mask = torch.tensor([True, False, True, True])

    selected = select_near_best_action(
        probabilities,
        tolerance=0.02,
        rng=np.random.default_rng(42),
        legal_mask=legal_mask,
    )

    assert selected in {0, 2}
    assert bool(legal_mask[selected])


def test_near_best_selection_rejects_invalid_tolerance():
    probabilities = torch.tensor([0.6, 0.4])

    with pytest.raises(ValueError, match="smaller than 1"):
        select_near_best_action(probabilities, tolerance=-0.01)

    with pytest.raises(ValueError, match="smaller than 1"):
        select_near_best_action(probabilities, tolerance=1.0)


def test_near_best_selection_rejects_mismatched_legal_mask():
    probabilities = torch.tensor([0.6, 0.4])
    legal_mask = torch.tensor([True, False, True])

    with pytest.raises(ValueError, match="same shape"):
        select_near_best_action(probabilities, legal_mask=legal_mask)


def test_apply_legal_move_mask_rejects_empty_legal_mask():
    logits = torch.tensor([1.0, 2.0, 3.0])
    legal_mask = torch.tensor([False, False, False])

    with pytest.raises(ValueError, match="No legal actions"):
        apply_legal_move_mask(logits, legal_mask)


def test_early_pass_penalty_only_applies_below_occupied_stone_threshold():
    logits = torch.zeros(82)
    logits[-1] = 2.0

    adjusted = apply_early_pass_penalty(logits, 9, 5, 3.5, 16)
    unchanged = apply_early_pass_penalty(logits, 9, 16, 3.5, 16)

    assert adjusted[-1] == pytest.approx(-1.5)
    assert logits[-1] == 2.0
    assert unchanged is logits


def test_early_pass_penalty_rejects_partial_configuration():
    with pytest.raises(ValueError, match="Set both"):
        apply_early_pass_penalty(torch.zeros(82), 9, 0, 3.5, 0)


def test_early_pass_penalty_accepts_both_supported_board_sizes():
    for board_size in (9, 19):
        logits = torch.zeros(board_size * board_size + 1)
        logits[-1] = 2.0
        adjusted = apply_early_pass_penalty(logits, board_size, 3, 3.5, 16)
        assert adjusted[-1] == pytest.approx(-1.5)


def test_agent_defaults_to_the_calibrated_early_pass_adjustment():
    parameters = inspect.signature(TRMAgent).parameters
    assert DEFAULT_PASS_LOGIT_PENALTY == 3.5
    assert DEFAULT_PASS_PENALTY_BEFORE_STONES == 16
    assert parameters["pass_logit_penalty"].default == 3.5
    assert parameters["pass_penalty_before_stones"].default == 16


def test_validation_accuracy_weights_each_example_equally():
    logits = torch.tensor(
        [[2.0, 0.0, -1.0], [0.0, 2.0, -1.0], [0.0, 2.0, -1.0]]
    )
    targets = torch.tensor([0, 0, 1])
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(logits, targets),
        batch_size=2,
        shuffle=False,
    )

    _, top1, top3 = evaluate(
        torch.nn.Identity(), loader, torch.nn.CrossEntropyLoss(), torch.device("cpu")
    )

    assert top1 == pytest.approx(200 / 3)
    assert top3 == pytest.approx(100.0)
