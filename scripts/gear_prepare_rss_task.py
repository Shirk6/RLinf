"""Prepare per-node RSS task assets inside a Gear Ray cluster."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import ray
from huggingface_hub import snapshot_download
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy


TASK_REPOS = {
    "tower": (
        ("Shirk6/wan-tower", "models/wan-tower"),
        (
            "Shirk6/pi05_tower-of-hanoi-game_weighted_bc_pytorch",
            "models/pi05_tower-of-hanoi-game_weighted_bc_pytorch",
        ),
    ),
    "battery": (
        ("Shirk6/wan-battery", "models/wan-battery"),
        (
            "Shirk6/pi05_insert-mouse-battery_weighted_bc_pytorch",
            "models/pi05_insert-mouse-battery_weighted_bc_pytorch",
        ),
    ),
    "bottle": (
        ("Shirk6/wan-bottle", "models/wan-bottle"),
        (
            "Shirk6/pi05_seal-water-bottle-cap_weighted_bc_pytorch",
            "models/pi05_seal-water-bottle-cap_weighted_bc_pytorch",
        ),
    ),
}


def _is_rate_limited(exc: BaseException) -> bool:
    """Return whether an exception looks like a Hugging Face rate limit."""
    response = getattr(exc, "response", None)
    if getattr(response, "status_code", None) == 429:
        return True
    message = str(exc)
    if "429" in message or "Too Many Requests" in message:
        return True
    return any(
        nested is not None and _is_rate_limited(nested)
        for nested in (getattr(exc, "__cause__", None), getattr(exc, "__context__", None))
    )


def snapshot_download_with_retries(
    *,
    snapshot_download_fn=snapshot_download,
    sleep_fn=time.sleep,
    max_attempts: int = 6,
    initial_sleep_seconds: int | None = None,
    **kwargs,
):
    """Download a Hugging Face snapshot with retries for transient rate limits."""
    sleep_seconds = (
        initial_sleep_seconds
        if initial_sleep_seconds is not None
        else int(os.environ.get("HF_RATE_LIMIT_SLEEP_SECONDS", "330"))
    )
    for attempt in range(1, max_attempts + 1):
        try:
            return snapshot_download_fn(**kwargs)
        except Exception as exc:
            if attempt == max_attempts or not _is_rate_limited(exc):
                raise
            print(
                f"Rate limited while downloading {kwargs.get('repo_id')}; "
                f"sleeping {sleep_seconds}s before retry {attempt + 1}/{max_attempts}.",
                flush=True,
            )
            sleep_fn(sleep_seconds)
            sleep_seconds *= 2
    raise RuntimeError("unreachable")


def _snapshot_download_with_retries(
    *,
    repo_id: str,
    target: Path,
    token: str | None,
    snapshot_download_fn=snapshot_download,
    max_attempts: int = 6,
) -> None:
    """Download a repo snapshot with conservative concurrency and 429 backoff."""
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    snapshot_download_with_retries(
        snapshot_download_fn=snapshot_download_fn,
        repo_id=repo_id,
        repo_type="model",
        local_dir=str(target),
        local_dir_use_symlinks=False,
        token=token,
        max_workers=1,
        max_attempts=max_attempts,
    )


def prepare_task_models(
    *,
    root: Path,
    task: str,
    token: str | None,
    skip_model_download: bool,
    snapshot_download_fn=snapshot_download,
) -> None:
    """Prepare task model paths by downloading or validating shared assets."""
    missing_paths = []
    for repo_id, local_dir in TASK_REPOS[task]:
        target = root / local_dir
        if skip_model_download:
            if not target.exists():
                missing_paths.append(str(target))
            continue

        target.mkdir(parents=True, exist_ok=True)
        _snapshot_download_with_retries(
            repo_id=repo_id,
            target=target,
            token=token,
            snapshot_download_fn=snapshot_download_fn,
        )

    if missing_paths:
        missing = "\n".join(f"- {path}" for path in missing_paths)
        raise RuntimeError(f"Missing required model paths:\n{missing}")


def _start_hf_watcher(
    *,
    root: Path,
    task: str,
    hf_repo_id: str | None,
    experiment_name: str | None,
    node_rank: int,
) -> str | None:
    """Start a detached checkpoint upload watcher on the current node."""
    if not hf_repo_id:
        return None
    if not experiment_name:
        raise RuntimeError("experiment_name is required when hf_repo_id is set.")

    hf_token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    )
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN, HUGGING_FACE_HUB_TOKEN, or HUGGINGFACE_HUB_TOKEN is required."
        )

    log_path = Path("/tmp") / f"hf_ckpt_watcher_{task}_{node_rank}.log"
    cmd = [
        sys.executable,
        "scripts/hf_upload_ckpt_watcher.py",
        "--task",
        task,
        "--experiment-name",
        experiment_name,
        "--repo-id",
        hf_repo_id,
        "--logs-root",
        str(root / "logs"),
        "--node-rank",
        str(node_rank),
    ]
    with log_path.open("ab") as log_file:
        subprocess.Popen(
            cmd,
            cwd=root,
            env=os.environ.copy(),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    return str(log_path)


@ray.remote(num_cpus=1)
def _prepare_node(
    task: str,
    hf_repo_id: str | None,
    experiment_name: str | None,
    node_rank: int,
    skip_model_download: bool,
) -> str:
    """Prepare task models and checkpoint upload watcher on the current Ray node."""
    root = Path.cwd()
    hf_token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    )
    prepare_task_models(
        root=root,
        task=task,
        token=hf_token,
        skip_model_download=skip_model_download,
    )

    watcher_log = _start_hf_watcher(
        root=root,
        task=task,
        hf_repo_id=hf_repo_id,
        experiment_name=experiment_name,
        node_rank=node_rank,
    )
    if watcher_log:
        print(f"Started HF checkpoint watcher on node {node_rank}: {watcher_log}")

    return ray.get_runtime_context().get_node_id()


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for RSS task preparation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=sorted(TASK_REPOS), required=True)
    parser.add_argument("--expected-nodes", type=int, default=8)
    parser.add_argument(
        "--node-wait-timeout-seconds",
        type=int,
        default=int(os.environ.get("RSS_PREPARE_NODE_WAIT_TIMEOUT_SECONDS", "1800")),
        help="Maximum time to wait for the expected Ray nodes to join.",
    )
    parser.add_argument(
        "--node-wait-poll-seconds",
        type=int,
        default=int(os.environ.get("RSS_PREPARE_NODE_WAIT_POLL_SECONDS", "10")),
        help="Polling interval while waiting for Ray nodes.",
    )
    parser.add_argument("--hf-repo-id", default=None)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument(
        "--node-delay-seconds",
        type=int,
        default=int(os.environ.get("RSS_PREPARE_NODE_DELAY_SECONDS", "120")),
    )
    parser.add_argument(
        "--max-concurrent-nodes",
        type=int,
        default=int(os.environ.get("RSS_PREPARE_MAX_CONCURRENT_NODES", "1")),
        help="Maximum number of Ray nodes to prepare concurrently.",
    )
    parser.add_argument(
        "--skip-model-download",
        action="store_true",
        help="Validate existing shared model paths instead of downloading them.",
    )
    return parser


def wait_for_ray_nodes(
    *,
    expected_nodes: int,
    timeout_seconds: int,
    poll_seconds: int,
    nodes_fn=ray.nodes,
    sleep_fn=time.sleep,
    monotonic_fn=time.monotonic,
) -> list[dict]:
    """Wait until the expected number of live Ray nodes is visible."""
    deadline = monotonic_fn() + timeout_seconds
    while True:
        nodes = [node for node in nodes_fn() if node.get("Alive")]
        if len(nodes) >= expected_nodes:
            return nodes

        now = monotonic_fn()
        if now >= deadline:
            msg = (
                f"Expected {expected_nodes} Ray nodes, found {len(nodes)} "
                f"after waiting {timeout_seconds}s."
            )
            raise RuntimeError(msg)

        sleep_seconds = min(poll_seconds, max(1, int(deadline - now)))
        print(
            f"Waiting for Ray nodes: found {len(nodes)}/{expected_nodes}; "
            f"sleeping {sleep_seconds}s.",
            flush=True,
        )
        sleep_fn(sleep_seconds)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.max_concurrent_nodes < 1:
        raise ValueError("--max-concurrent-nodes must be at least 1.")
    if args.node_wait_timeout_seconds < 0:
        raise ValueError("--node-wait-timeout-seconds must be non-negative.")
    if args.node_wait_poll_seconds < 1:
        raise ValueError("--node-wait-poll-seconds must be at least 1.")

    ray.init(address="auto")
    nodes = wait_for_ray_nodes(
        expected_nodes=args.expected_nodes,
        timeout_seconds=args.node_wait_timeout_seconds,
        poll_seconds=args.node_wait_poll_seconds,
    )

    prepared = []
    pending = []
    nodes_to_prepare = nodes[: args.expected_nodes]
    for node_rank, node in enumerate(nodes_to_prepare):
        ref = _prepare_node.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=node["NodeID"],
                soft=False,
            )
        ).remote(
            args.task,
            args.hf_repo_id,
            args.experiment_name,
            node_rank,
            args.skip_model_download,
        )
        pending.append((node_rank, ref))
        if len(pending) >= args.max_concurrent_nodes:
            prepared_node_rank, prepared_ref = pending.pop(0)
            prepared.append(ray.get(prepared_ref))
            print(f"Prepared node {prepared_node_rank}.", flush=True)
        elif args.node_delay_seconds > 0 and node_rank < len(nodes_to_prepare) - 1:
            print(
                f"Scheduled node {node_rank}; sleeping "
                f"{args.node_delay_seconds}s before scheduling next HF download.",
                flush=True,
            )
            time.sleep(args.node_delay_seconds)

    for node_rank, ref in pending:
        prepared.append(ray.get(ref))
        print(f"Prepared node {node_rank}.", flush=True)

    if len(set(prepared)) != args.expected_nodes:
        msg = f"Prepared {len(set(prepared))} unique nodes, expected {args.expected_nodes}."
        raise RuntimeError(msg)

    print(f"Prepared {args.task} assets on {len(set(prepared))} Ray nodes.")


if __name__ == "__main__":
    main()
