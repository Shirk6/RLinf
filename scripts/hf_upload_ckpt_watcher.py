#!/usr/bin/env python3
"""Upload RLinf checkpoints to Hugging Face as they are written."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi


STEP_RE = re.compile(r"^global_step_(\d+)$")


def log(message: str) -> None:
    """Print a timestamped watcher log line."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def load_state(path: Path) -> dict[str, Any]:
    """Load uploaded-checkpoint state."""
    if not path.exists():
        return {"uploaded": []}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {"uploaded": []}


def save_state(path: Path, state: dict[str, Any]) -> None:
    """Persist uploaded-checkpoint state atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(state, indent=2, sort_keys=True))
    tmp_path.replace(path)


def checkpoint_snapshot(path: Path) -> tuple[int, int, float]:
    """Return file count, total size, and latest mtime for a directory tree."""
    file_count = 0
    total_size = 0
    latest_mtime = path.stat().st_mtime
    for item in path.rglob("*"):
        try:
            stat = item.stat()
        except FileNotFoundError:
            continue
        latest_mtime = max(latest_mtime, stat.st_mtime)
        if item.is_file():
            file_count += 1
            total_size += stat.st_size
    return file_count, total_size, latest_mtime


def iter_checkpoint_dirs(logs_root: Path, experiment_name: str) -> list[Path]:
    """Find checkpoint directories for the experiment under all run log dirs."""
    candidates: list[tuple[int, Path]] = []
    for checkpoint_dir in logs_root.glob(
        f"*/{experiment_name}/checkpoints/global_step_*"
    ):
        if not checkpoint_dir.is_dir():
            continue
        match = STEP_RE.match(checkpoint_dir.name)
        if match is None:
            continue
        candidates.append((int(match.group(1)), checkpoint_dir))
    candidates.sort(key=lambda item: (item[0], str(item[1])))
    return [path for _, path in candidates]


def upload_with_retries(
    api: HfApi,
    repo_id: str,
    folder_path: Path,
    path_in_repo: str,
    token: str,
    max_attempts: int,
) -> None:
    """Upload a checkpoint folder, retrying transient commit conflicts."""
    for attempt in range(1, max_attempts + 1):
        try:
            api.upload_folder(
                repo_id=repo_id,
                repo_type="model",
                folder_path=str(folder_path),
                path_in_repo=path_in_repo,
                token=token,
                commit_message=f"Upload {path_in_repo}",
                ignore_patterns=["*.tmp", "*.lock", ".nfs*"],
            )
            return
        except Exception as exc:  # noqa: BLE001 - keep watcher alive on hub errors.
            if attempt == max_attempts:
                raise
            sleep_seconds = min(300, 30 * attempt)
            log(
                "Upload failed for "
                f"{path_in_repo} on attempt {attempt}/{max_attempts}: "
                f"{type(exc).__name__}. Retrying in {sleep_seconds}s."
            )
            time.sleep(sleep_seconds)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--logs-root", type=Path, default=Path("logs"))
    parser.add_argument("--node-rank", required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--stable-seconds", type=int, default=60)
    parser.add_argument("--max-upload-attempts", type=int, default=5)
    parser.add_argument("--state-path", type=Path, default=None)
    args = parser.parse_args()

    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    )
    if not token:
        log("HF_TOKEN, HUGGING_FACE_HUB_TOKEN, or HUGGINGFACE_HUB_TOKEN is required.")
        return 2

    logs_root = args.logs_root.resolve()
    state_path = args.state_path
    if state_path is None:
        state_path = Path("/tmp") / f"hf_uploaded_ckpts_{args.task}_{args.node_rank}.json"

    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=False,
        exist_ok=True,
        token=token,
    )

    log(
        "Watching "
        f"{logs_root} for {args.experiment_name} checkpoints on node "
        f"{args.node_rank}; uploading to {args.repo_id}."
    )

    state = load_state(state_path)
    uploaded = set(state.get("uploaded", []))

    while True:
        for checkpoint_dir in iter_checkpoint_dirs(logs_root, args.experiment_name):
            try:
                file_count, total_size, latest_mtime = checkpoint_snapshot(
                    checkpoint_dir
                )
            except FileNotFoundError:
                continue
            if file_count == 0:
                continue
            if time.time() - latest_mtime < args.stable_seconds:
                continue

            relative_path = checkpoint_dir.resolve().relative_to(logs_root)
            path_in_repo = str(
                Path("runs") / relative_path / f"node_{args.node_rank}"
            )
            upload_key = f"{path_in_repo}:{file_count}:{total_size}"
            if upload_key in uploaded:
                continue

            log(f"Uploading {checkpoint_dir} to {args.repo_id}/{path_in_repo}.")
            try:
                upload_with_retries(
                    api=api,
                    repo_id=args.repo_id,
                    folder_path=checkpoint_dir,
                    path_in_repo=path_in_repo,
                    token=token,
                    max_attempts=args.max_upload_attempts,
                )
            except Exception as exc:  # noqa: BLE001 - continue watching.
                log(
                    f"Upload failed for {path_in_repo}: "
                    f"{type(exc).__name__}: {exc}"
                )
                continue

            uploaded.add(upload_key)
            state["uploaded"] = sorted(uploaded)
            save_state(state_path, state)
            log(f"Uploaded {path_in_repo}.")

        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    sys.exit(main())
