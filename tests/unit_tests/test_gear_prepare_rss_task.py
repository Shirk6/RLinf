"""Tests for RSS Gear launch preparation helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_prepare_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "gear_prepare_rss_task.py"
    spec = importlib.util.spec_from_file_location("gear_prepare_rss_task", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_script_does_not_read_wandb_api_key() -> None:
    """Gear launch should rely on existing wandb login instead of API key env."""
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "gear_run_rss_task.sh"

    assert "WANDB_API_KEY" not in script.read_text(encoding="utf-8")


def test_tower_task_downloads_tower_models() -> None:
    """Tower launch preparation should download Tower Wan and pi0.5 assets."""
    module = _load_prepare_module()

    assert module.TASK_REPOS["tower"] == (
        ("Shirk6/wan-tower", "models/wan-tower"),
        (
            "Shirk6/pi05_tower-of-hanoi-game_weighted_bc_pytorch",
            "models/pi05_tower-of-hanoi-game_weighted_bc_pytorch",
        ),
    )


def test_snapshot_download_retries_transient_errors() -> None:
    """Transient Hub failures should be retried before failing preparation."""
    module = _load_prepare_module()
    calls = []
    sleeps = []

    def flaky_snapshot_download(**kwargs):
        calls.append(kwargs)
        if len(calls) < 3:
            raise RuntimeError("429 Client Error: Too Many Requests")
        return "/tmp/model"

    result = module.snapshot_download_with_retries(
        snapshot_download_fn=flaky_snapshot_download,
        sleep_fn=sleeps.append,
        max_attempts=3,
        initial_sleep_seconds=30,
        repo_id="Shirk6/wan-tower",
        repo_type="model",
        local_dir="/tmp/model",
    )

    assert result == "/tmp/model"
    assert len(calls) == 3
    assert sleeps == [30, 60]


def test_parser_accepts_bounded_node_concurrency() -> None:
    """RSS task preparation should expose opt-in bounded node concurrency."""
    module = _load_prepare_module()

    args = module.build_parser().parse_args(
        [
            "--task",
            "tower",
            "--max-concurrent-nodes",
            "4",
        ]
    )

    assert args.max_concurrent_nodes == 4


def test_parser_can_skip_model_downloads_for_shared_filesystem() -> None:
    """RSS task preparation should support shared filesystems without downloads."""
    module = _load_prepare_module()

    args = module.build_parser().parse_args(
        [
            "--task",
            "tower",
            "--skip-model-download",
        ]
    )

    assert args.skip_model_download is True


def test_prepare_task_models_skips_download_when_paths_exist(tmp_path) -> None:
    """Shared filesystem preparation should validate paths instead of downloading."""
    module = _load_prepare_module()
    for _, local_dir in module.TASK_REPOS["tower"]:
        (tmp_path / local_dir).mkdir(parents=True)

    def unexpected_download(**_kwargs):
        raise AssertionError("snapshot_download should not be called")

    module.prepare_task_models(
        root=tmp_path,
        task="tower",
        token=None,
        skip_model_download=True,
        snapshot_download_fn=unexpected_download,
    )


def test_wait_for_ray_nodes_polls_until_expected_nodes_join() -> None:
    """Preparation should tolerate workers joining Ray after the master."""
    module = _load_prepare_module()
    calls = []
    sleeps = []

    def nodes_fn():
        calls.append(None)
        node_count = 1 if len(calls) == 1 else 8
        return [{"Alive": True, "NodeID": str(index)} for index in range(node_count)]

    nodes = module.wait_for_ray_nodes(
        expected_nodes=8,
        timeout_seconds=60,
        poll_seconds=10,
        nodes_fn=nodes_fn,
        sleep_fn=sleeps.append,
        monotonic_fn=lambda: len(calls),
    )

    assert len(nodes) == 8
    assert sleeps == [10]


def test_wait_for_ray_nodes_times_out_when_nodes_do_not_join() -> None:
    """Preparation should still fail clearly if the Ray cluster never fills."""
    module = _load_prepare_module()
    clock = [0]

    def sleep_fn(seconds):
        clock[0] += seconds

    try:
        module.wait_for_ray_nodes(
            expected_nodes=8,
            timeout_seconds=5,
            poll_seconds=2,
            nodes_fn=lambda: [{"Alive": True, "NodeID": "head"}],
            sleep_fn=sleep_fn,
            monotonic_fn=lambda: clock[0],
        )
    except RuntimeError as exc:
        assert "Expected 8 Ray nodes, found 1 after waiting 5s." in str(exc)
    else:
        raise AssertionError("wait_for_ray_nodes should time out")
