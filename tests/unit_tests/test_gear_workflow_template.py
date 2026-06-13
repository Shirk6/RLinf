"""Tests for Gear workflow template safety properties."""

from __future__ import annotations

from pathlib import Path


def test_ray_job_submit_disables_xtrace_before_runtime_env_json() -> None:
    """Runtime env JSON may contain secrets and must not be echoed by set -x."""
    repo_root = Path(__file__).resolve().parents[2]
    template = repo_root / "gear_workflows" / "ray_cluster_rlinf.yaml"
    lines = template.read_text(encoding="utf-8").splitlines()

    submit_indexes = [
        index for index, line in enumerate(lines) if "ray job submit" in line
    ]

    assert submit_indexes
    for index in submit_indexes:
        preceding = "\n".join(lines[max(0, index - 8) : index])
        assert "set +x" in preceding
