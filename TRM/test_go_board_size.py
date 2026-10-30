import numpy as np
import pytest
import torch

from go_dataset import apply_board_symmetry
from app import checkpoint_for_board_size
from train import GoTRMModel, evaluate
from agents import select_near_best_action


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


def test_near_best_selection_randomizes_only_within_tolerance():
    probabilities = torch.tensor([0.51, 0.50, 0.48, 0.0])
    rng = np.random.default_rng(42)

    selected = {
        select_near_best_action(probabilities, tolerance=0.02, rng=rng)
        for _ in range(100)
    }

    assert selected == {0, 1}


def test_near_best_selection_keeps_clear_best_action():
    probabilities = torch.tensor([0.80, 0.77, 0.0])

    assert select_near_best_action(
        probabilities, tolerance=0.02, rng=np.random.default_rng(42)
    ) == 0


def test_near_best_selection_rejects_negative_tolerance():
    with pytest.raises(ValueError, match="non-negative"):
        select_near_best_action(torch.tensor([0.6, 0.4]), tolerance=-0.01)


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
