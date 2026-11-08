import os
import shutil
import tempfile
import zipfile
from pathlib import Path

import pytest

from game_env import GoGameEnv
from model_evaluation.go_match import (
    GTPClient,
    configure_katago,
    gtp_to_action,
    load_board_rules,
)


TRM_DIR = Path(__file__).parents[1]
KATAGO_DIR = TRM_DIR / "model_evaluation" / "katago-local"
KATAGO_ENABLED = os.environ.get("KATAGO_INTEGRATION_TESTS") == "1"


def _local_resources(board_size):
    executable_override = os.environ.get("KATAGO_EXECUTABLE")
    executable_candidates = [
        Path(executable_override) if executable_override else None,
        KATAGO_DIR / "engine" / "squashfs-root" / "usr" / "bin" / "katago",
        Path(shutil.which("katago")) if shutil.which("katago") else None,
    ]
    executable = next(
        (path for path in executable_candidates if path is not None and path.is_file()),
        None,
    )

    config_override = os.environ.get("KATAGO_CONFIG")
    config = (
        Path(config_override)
        if config_override
        else KATAGO_DIR / "engine" / "default_gtp.cfg"
    )
    if board_size == 9:
        model_name = "kata9x9-b18c384nbt-20231025.bin.gz"
    else:
        model_name = "kata1-tf3-b11c768-s12382M-d6533M.bin.gz"
    model_override = os.environ.get(f"KATAGO_MODEL_{board_size}")
    model = (
        Path(model_override)
        if model_override
        else KATAGO_DIR / "downloads" / model_name
    )

    if executable is None or not config.is_file() or not model.is_file():
        return None
    return executable, config, model


@pytest.mark.skipif(
    not KATAGO_ENABLED,
    reason="Set KATAGO_INTEGRATION_TESTS=1 to run the local KataGo engine.",
)
@pytest.mark.parametrize("board_size", [9, 19])
def test_local_katago_rules_moves_and_pass_termination(board_size, tmp_path, monkeypatch):
    resources = _local_resources(board_size)
    if resources is None:
        pytest.skip(f"Local KataGo executable, config, or {board_size}x{board_size} net is missing.")
    executable, config, model = resources

    library_paths = []
    appimage_libraries = KATAGO_DIR / "engine" / "squashfs-root" / "usr" / "lib"
    if appimage_libraries.is_dir():
        library_paths.append(str(appimage_libraries))

    cudnn_wheel = KATAGO_DIR / "downloads" / "nvidia_cudnn_cu13-9.28.1.4.whl"
    with tempfile.TemporaryDirectory(prefix="katago-cudnn-", dir=tmp_path) as cudnn_dir:
        if cudnn_wheel.is_file():
            with zipfile.ZipFile(cudnn_wheel) as wheel:
                libraries = [
                    name
                    for name in wheel.namelist()
                    if name.startswith("nvidia/cudnn/lib/") and ".so" in name
                ]
                for name in libraries:
                    Path(cudnn_dir, Path(name).name).write_bytes(wheel.read(name))
            library_paths.insert(0, cudnn_dir)

        existing_library_path = os.environ.get("LD_LIBRARY_PATH", "")
        library_path = ":".join(
            [*library_paths, existing_library_path] if existing_library_path else library_paths
        )
        if library_path:
            monkeypatch.setenv("LD_LIBRARY_PATH", library_path)

        komi, rules = load_board_rules(
            TRM_DIR / "model_evaluation" / "config" / "rules.json",
            board_size,
        )
        command = [
            str(executable),
            "gtp",
            "-config",
            str(config),
            "-model",
            str(model),
            "-override-config",
            (
                f"maxVisits=1,numSearchThreads=1,logDir={tmp_path},"
                "delayMoveScale=0,delayMoveMax=0"
            ),
        ]
        client = GTPClient(command, timeout=180)
        try:
            assert client.command("protocol_version") == "2"
            assert client.command("name") == "KataGo"
            configure_katago(client, board_size, komi, rules)
            assert client.command("kata-get-rules")
            assert float(client.command("get_komi")) == pytest.approx(komi)

            env = GoGameEnv(board_size=board_size, komi=komi)
            generated_move = client.command("genmove b").strip()
            action = gtp_to_action(generated_move, board_size)
            assert env.is_legal_move(action) == (True, None)
            assert not env.step(action)

            first_pass = board_size**2
            assert client.command("play w pass") == ""
            assert not env.step(first_pass)
            assert env.current_player == "b"
            assert env.consecutive_passes == 1

            assert client.command("play b pass") == ""
            assert env.step(first_pass)
            assert env.game_over
            assert env.consecutive_passes == 2
            assert client.command("final_score")
        finally:
            client.close()
