from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pytest
from sgfmill import sgf

from go_dataset import GoDataset, default_go_dataset_dir
from train import resolve_dataset_paths
from model_evaluation.go_eval import default_test_data_path
from go_preprocessing import (
    build_dataset,
    build_memmap_dataset,
    count_sgf_examples,
    default_sgf_dir,
    process_sgf_file,
)


EXTRACTOR_PATH = (
    Path(__file__).parent.parent
    / "9x9_go"
    / "go_raw_dataset"
    / "extract_9x9.py"
)
EXTRACTOR_SPEC = spec_from_file_location("extract_go_games", EXTRACTOR_PATH)
EXTRACTOR_MODULE = module_from_spec(EXTRACTOR_SPEC)
EXTRACTOR_SPEC.loader.exec_module(EXTRACTOR_MODULE)


def _sgf_bytes(board_size, moves=(), result=None):
    game = sgf.Sgf_game(size=board_size)
    if result is not None:
        game.get_root().set("RE", result)
    node = game.get_root()
    for color, coordinate in moves:
        node = node.new_child()
        if coordinate is None:
            node.set(color.upper(), None)
        else:
            node.set(color.upper(), coordinate)
    return game.serialise()


def test_extractor_filters_nested_sources_and_deduplicates(tmp_path):
    source_a = tmp_path / "source-a"
    source_b = tmp_path / "source-b"
    source_a.mkdir()
    (source_a / "nested").mkdir()
    source_b.mkdir()
    nineteen = _sgf_bytes(19)
    (source_a / "nested" / "game.SGF").write_bytes(nineteen)
    (source_a / "nine.sgf").write_bytes(_sgf_bytes(9))
    (source_b / "duplicate.sgf").write_bytes(nineteen)
    (source_b / "broken.sgf").write_bytes(b"not an SGF")
    destination = tmp_path / "filtered"

    summary = EXTRACTOR_MODULE.extract_games(
        [source_a, source_b], destination, board_size=19
    )

    extracted = list(destination.glob("*.sgf"))
    assert summary["analyzed"] == 4
    assert summary["copied"] == 1
    assert summary["duplicates"] == 1
    assert summary["wrong_size"] == 1
    assert summary["unreadable"] == 1
    assert len(extracted) == 1
    assert sgf.Sgf_game.from_bytes(extracted[0].read_bytes()).get_size() == 19


@pytest.mark.parametrize(
    ("board_size", "directory_name"),
    [(9, "go9_filtered"), (19, "19x19_professional_games")],
)
def test_extractor_defaults_to_documented_filtered_directory(
    tmp_path, board_size, directory_name
):
    destination = EXTRACTOR_MODULE.default_destination_dir(board_size, tmp_path)

    assert destination == (
        tmp_path
        / f"{board_size}x{board_size}_go"
        / "go_raw_dataset"
        / directory_name
    )


def test_count_matches_processed_moves_and_reconstructed_passes(tmp_path):
    path = tmp_path / "scored.sgf"
    path.write_bytes(
        _sgf_bytes(
            9,
            moves=[("b", (0, 0)), ("w", (1, 1))],
            result="B+2.5",
        )
    )

    features, targets, metadata = process_sgf_file(path, board_size=9)

    assert count_sgf_examples(path, board_size=9) == len(targets) == 4
    assert features[0].shape == (7, 9, 9)
    assert features[0][6].all()
    assert features[1][1, 0, 0] == 1
    assert features[1][0, 0, 0] == 0
    assert metadata["added_terminal_passes"] == 2
    assert targets[-2:] == [81, 81]


def test_count_does_not_reconstruct_passes_after_an_illegal_move(tmp_path):
    path = tmp_path / "illegal-scored.sgf"
    path.write_bytes(
        _sgf_bytes(
            9,
            moves=[("b", (0, 0)), ("w", (0, 0))],
            result="B+2.5",
        )
    )

    _, targets, metadata = process_sgf_file(path, board_size=9)

    assert metadata["status"] == "invalid_move"
    assert count_sgf_examples(path, board_size=9) == len(targets) == 1


@pytest.mark.parametrize("board_size", [9, 19])
def test_build_dataset_writes_standard_npz_in_bounded_batches(
    tmp_path,
    board_size,
):
    sgf_paths = []
    for index in range(3):
        path = tmp_path / f"game-{index}.sgf"
        path.write_bytes(
            _sgf_bytes(
                board_size,
                moves=[("b", (index, index)), ("w", (index, index + 1))],
                result="W+R",
            )
        )
        sgf_paths.append(path)

    expected_examples = sum(
        count_sgf_examples(path, board_size) for path in sgf_paths
    )
    output_path = tmp_path / "dataset.npz"
    summary = build_dataset(
        sgf_paths,
        output_path,
        board_size=board_size,
        expected_examples=expected_examples,
    )

    with np.load(output_path) as dataset:
        assert dataset["X"].shape == (
            expected_examples,
            7,
            board_size,
            board_size,
        )
        assert dataset["X"].dtype == np.float32
        assert dataset["y"].shape == (expected_examples,)
        assert dataset["y"].dtype == np.int64
        assert np.all(dataset["y"] < board_size**2)
    sample_x, sample_y = GoDataset(output_path, board_size=board_size)[0]
    assert tuple(sample_x.shape) == (7, board_size, board_size)
    assert 0 <= sample_y.item() < board_size**2
    assert summary["examples"] == expected_examples


@pytest.mark.parametrize("board_size", [9, 19])
def test_memmap_dataset_is_loadable_without_loading_split_into_memory(
    tmp_path,
    board_size,
):
    sgf_paths = []
    for index in range(3):
        path = tmp_path / f"memmap-game-{index}.sgf"
        path.write_bytes(
            _sgf_bytes(
                board_size,
                moves=[("b", (index, index)), ("w", (index, index + 1))],
                result="W+R",
            )
        )
        sgf_paths.append(path)

    expected_examples = sum(
        count_sgf_examples(path, board_size) for path in sgf_paths
    )
    output_dir = tmp_path / "train"
    summary = build_memmap_dataset(
        sgf_paths,
        output_dir,
        board_size=board_size,
        expected_examples=expected_examples,
    )

    dataset = GoDataset(output_dir, board_size=board_size)
    sample_x, sample_y = dataset[0]
    assert isinstance(dataset.X, np.memmap)
    assert dataset.X.shape == (
        expected_examples,
        7,
        board_size,
        board_size,
    )
    assert dataset.y.shape == (expected_examples,)
    assert tuple(sample_x.shape) == (7, board_size, board_size)
    assert 0 <= sample_y.item() <= board_size**2
    assert summary["examples"] == expected_examples


def test_training_auto_format_prefers_memory_mapped_splits(tmp_path):
    npy_train = tmp_path / "train"
    npy_val = tmp_path / "val"
    npy_train.mkdir()
    npy_val.mkdir()
    npz_train = tmp_path / "train.npz"
    npz_val = tmp_path / "val.npz"
    npz_train.touch()
    npz_val.touch()

    assert resolve_dataset_paths(tmp_path) == (npy_train, npy_val)
    assert resolve_dataset_paths(tmp_path, "npz") == (npz_train, npz_val)


@pytest.mark.parametrize(
    ("board_size", "dataset_name", "input_name"),
    [
        (9, "go_dataset_go9", "go9_filtered"),
        (19, "go_dataset_professional_10k", "19x19_professional_games"),
    ],
)
def test_default_dataset_paths_match_board_size_layout(
    tmp_path, board_size, dataset_name, input_name
):
    assert default_go_dataset_dir(board_size, tmp_path) == (
        tmp_path / f"{board_size}x{board_size}_go" / dataset_name
    )
    assert default_sgf_dir(board_size, tmp_path) == (
        tmp_path / f"{board_size}x{board_size}_go" / "go_raw_dataset" / input_name
    )


def test_default_evaluation_split_prefers_memory_mapped_data(tmp_path, monkeypatch):
    import model_evaluation.go_eval as go_eval

    monkeypatch.setattr(go_eval, "SCRIPT_DIR", tmp_path)
    dataset_dir = tmp_path / "19x19_go" / "go_dataset_professional_10k"
    npy_test = dataset_dir / "test"
    npy_test.mkdir(parents=True)
    (dataset_dir / "test.npz").touch()

    assert default_test_data_path(19) == npy_test

    npy_test.rmdir()
    assert default_test_data_path(19) == dataset_dir / "test.npz"


def test_default_nine_by_nine_evaluation_uses_documented_dataset_directory(
    tmp_path, monkeypatch
):
    import model_evaluation.go_eval as go_eval

    monkeypatch.setattr(go_eval, "SCRIPT_DIR", tmp_path)
    dataset_dir = tmp_path / "9x9_go" / "go_dataset_go9"
    npy_test = dataset_dir / "test"
    npy_test.mkdir(parents=True)

    assert default_test_data_path(9) == npy_test
