import numpy as np
import pytest
import torch
import torch.nn as nn

import train as trainer


class _TinyGoModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.output = nn.Linear(1, config["vocab_size"])

    def forward(self, inputs):
        features = inputs.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(-1)
        return self.output(features)


def _write_dataset(dataset_dir, board_size):
    rng = np.random.default_rng(42)
    for split, count in (("train", 6), ("val", 4)):
        features = rng.normal(
            size=(count, 7, board_size, board_size)
        ).astype(np.float32)
        targets = rng.integers(0, board_size**2 + 1, size=count)
        np.savez(dataset_dir / f"{split}.npz", X=features, y=targets)


@pytest.mark.parametrize("board_size", [9, 19])
def test_resume_continues_exactly_from_saved_batch_and_epoch(
    tmp_path, monkeypatch, board_size
):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    _write_dataset(dataset_dir, board_size)

    full_trm_dir = tmp_path / "full" / "TRM"
    resumed_trm_dir = tmp_path / "resumed" / "TRM"
    full_trm_dir.mkdir(parents=True)
    resumed_trm_dir.mkdir(parents=True)
    monkeypatch.setattr(trainer, "GoTRMModel", _TinyGoModel)

    common = {
        "board_size": board_size,
        "dataset_dir": dataset_dir,
        "dataset_format": "npz",
        "batch_size": 2,
        "num_workers": 0,
        "epochs": 2,
        "learning_rate": 0.01,
        "min_learning_rate": 0.001,
        "warmup_steps": 0,
        "weight_decay": 0.0,
        "checkpoint_every_steps": 1,
        "seed": 7,
    }

    monkeypatch.setattr(trainer, "__file__", str(full_trm_dir / "train.py"))
    trainer.train(**common)
    full_checkpoint_path = (
        full_trm_dir
        / "checkpoints"
        / (f"{board_size}x{board_size}_go" if board_size != 9 else "")
        / "latest_model.pt"
    )
    full_checkpoint = torch.load(full_checkpoint_path, map_location="cpu")

    monkeypatch.setattr(trainer, "__file__", str(resumed_trm_dir / "train.py"))
    trainer.train(**common, max_steps=1)
    interrupted_path = (
        resumed_trm_dir
        / "checkpoints"
        / (f"{board_size}x{board_size}_go" if board_size != 9 else "")
        / "latest_model.pt"
    )
    interrupted = torch.load(interrupted_path, map_location="cpu")
    assert interrupted["resume_epoch"] == 1
    assert interrupted["step"] == 1
    assert interrupted["seed"] == 7
    assert interrupted["rng_state"]["train_generator"].numel() > 0
    assert interrupted["optimizer_config"]["name"] == "AdamW"
    assert interrupted["scheduler_config"]["type"] == trainer.SCHEDULER_TYPE
    assert interrupted["scheduler_config"]["total_steps"] == 6
    assert interrupted["dataset_info"]["train"]["num_examples"] == 6

    trainer.train(
        board_size=board_size,
        epochs=2,
        resume=True,
    )
    resumed = torch.load(interrupted_path, map_location="cpu")

    assert resumed["resume_epoch"] == 3
    assert resumed["completed_epoch"] == 2
    assert resumed["global_step"] == full_checkpoint["global_step"] == 6
    for name, value in full_checkpoint["model_state_dict"].items():
        assert torch.equal(value, resumed["model_state_dict"][name])

    legacy_checkpoint = dict(resumed)
    for key in (
        "resume_epoch",
        "completed_epoch",
        "seed",
        "training_config",
        "optimizer_config",
        "scheduler_config",
        "dataset_info",
        "rng_state",
    ):
        legacy_checkpoint.pop(key, None)
    torch.save(legacy_checkpoint, interrupted_path)

    trainer.train(
        board_size=board_size,
        dataset_dir=dataset_dir,
        dataset_format="npz",
        batch_size=2,
        num_workers=0,
        epochs=3,
        learning_rate=0.01,
        min_learning_rate=0.001,
        warmup_steps=0,
        weight_decay=0.0,
        checkpoint_every_steps=1,
        resume=True,
    )
    legacy_resumed = torch.load(interrupted_path, map_location="cpu")
    assert legacy_resumed["completed_epoch"] == 3
    assert legacy_resumed["global_step"] == 9
    assert legacy_resumed["seed"] == 42
