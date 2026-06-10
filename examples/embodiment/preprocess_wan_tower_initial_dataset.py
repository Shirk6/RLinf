#!/usr/bin/env python3
"""Build Wan Tower initial-state trajectories from LeRobot expert data.

The Wan Tower environment reads one ``.npy`` trajectory per reset state through
``NpyTrajectoryDatasetWrapper``.  This script converts the Challenge phase-1
Tower of Hanoi expert dataset into that format:

* first 25 source frames are considered;
* condition images use source frame indices [0, 6, 12, 18, 24];
* target condition actions use [5, 11, 17, 23], matching Wan's stride-6
  action downsampling;
* the three camera views are stacked vertically as high, left wrist, right
  wrist, then padded from 540x320 to 544x320.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import imageio_ffmpeg
import numpy as np
import pandas as pd


CAMERA_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)
DEFAULT_FRAME_INDICES = (0, 5, 11, 17, 23)
DEFAULT_ACTION_INDICES = (0, 5, 11, 17, 23)
TASK = "tower-of-hanoi-game"


@dataclass(frozen=True)
class EpisodeJob:
    episode_index: int
    length: int
    source_root: Path
    output_dir: Path
    frame_indices: tuple[int, ...]
    action_indices: tuple[int, ...]
    view_height: int
    width: int
    padding_bottom: int
    chunk_size: int
    overwrite: bool
    ffmpeg_exe: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Tower of Hanoi expert data into Wan initial-state npy trajectories."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("Challenge-phase1-dataset/tower-of-hanoi-game/expert-data"),
        help="LeRobot-format expert-data root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/project/peilab/srk/rss_2026_ws/models/wan-tower/dataset"),
        help="Output directory consumed by env.train.initial_image_path.",
    )
    parser.add_argument(
        "--frame-indices",
        default=",".join(map(str, DEFAULT_FRAME_INDICES)),
        help="Comma-separated source frame indices used as condition images.",
    )
    parser.add_argument(
        "--action-indices",
        default=",".join(map(str, DEFAULT_ACTION_INDICES)),
        help=(
            "Comma-separated source action indices for the saved frames. "
            "The first one is unused by Wan reset; target items use the last four."
        ),
    )
    parser.add_argument("--view-height", type=int, default=180)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--padding-bottom", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Optional cap for smoke tests.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate existing npy files.",
    )
    return parser.parse_args()


def parse_int_tuple(value: str) -> tuple[int, ...]:
    items = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not items:
        raise ValueError("Expected at least one integer")
    return items


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def episode_chunk(episode_index: int, chunk_size: int) -> int:
    return episode_index // chunk_size


def parquet_path(source_root: Path, episode_index: int, chunk_size: int) -> Path:
    return (
        source_root
        / "data"
        / f"chunk-{episode_chunk(episode_index, chunk_size):03d}"
        / f"episode_{episode_index:06d}.parquet"
    )


def video_paths(source_root: Path, episode_index: int, chunk_size: int) -> list[Path]:
    chunk = episode_chunk(episode_index, chunk_size)
    return [
        source_root
        / "videos"
        / f"chunk-{chunk:03d}"
        / camera
        / f"episode_{episode_index:06d}.mp4"
        for camera in CAMERA_KEYS
    ]


def build_ffmpeg_filter(frame_indices: tuple[int, ...]) -> str:
    select_expr = "+".join(f"eq(n\\,{idx})" for idx in frame_indices)
    filter_parts = [
        f"[{stream_idx}:v]select={select_expr},setpts=N/FRAME_RATE/TB[v{stream_idx}]"
        for stream_idx in range(len(CAMERA_KEYS))
    ]
    stacked = "".join(f"[v{stream_idx}]" for stream_idx in range(len(CAMERA_KEYS)))
    filter_parts.append(f"{stacked}vstack=inputs={len(CAMERA_KEYS)},format=rgb24[out]")
    return ";".join(filter_parts)


def extract_stacked_frames(job: EpisodeJob) -> np.ndarray:
    paths = video_paths(job.source_root, job.episode_index, job.chunk_size)
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing video files: {missing}")

    cmd = [job.ffmpeg_exe, "-nostdin", "-v", "error", "-threads", "1"]
    for path in paths:
        cmd.extend(["-i", str(path)])
    cmd.extend(
        [
            "-filter_complex",
            build_ffmpeg_filter(job.frame_indices),
            "-map",
            "[out]",
            "-vsync",
            "0",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ]
    )

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed for episode {job.episode_index}: {proc.stderr.decode('utf-8', 'replace')}"
        )

    stacked_height = job.view_height * len(CAMERA_KEYS)
    expected_bytes = len(job.frame_indices) * stacked_height * job.width * 3
    if len(proc.stdout) != expected_bytes:
        raise RuntimeError(
            f"Unexpected decoded byte count for episode {job.episode_index}: "
            f"{len(proc.stdout)} != {expected_bytes}"
        )

    frames = np.frombuffer(proc.stdout, dtype=np.uint8).reshape(
        len(job.frame_indices), stacked_height, job.width, 3
    )
    if job.padding_bottom:
        frames = np.pad(
            frames,
            ((0, 0), (0, job.padding_bottom), (0, 0), (0, 0)),
            mode="constant",
            constant_values=0,
        )
    return frames


def load_states_actions(job: EpisodeJob) -> tuple[list[np.ndarray], list[np.ndarray]]:
    path = parquet_path(job.source_root, job.episode_index, job.chunk_size)
    if not path.exists():
        raise FileNotFoundError(f"Missing parquet file: {path}")

    required_until = max(max(job.frame_indices), max(job.action_indices)) + 1
    df = pd.read_parquet(path, columns=["observation.state", "action"])
    if len(df) < required_until:
        raise ValueError(
            f"Episode {job.episode_index} has {len(df)} rows, need at least {required_until}"
        )

    states = [
        np.asarray(df.iloc[frame_idx]["observation.state"], dtype=np.float32)
        for frame_idx in job.frame_indices
    ]
    actions = [
        np.asarray(df.iloc[action_idx]["action"], dtype=np.float32)
        for action_idx in job.action_indices
    ]
    return states, actions


def convert_episode(job: EpisodeJob) -> tuple[int, str]:
    out_path = job.output_dir / f"episode_{job.episode_index:06d}.npy"
    if out_path.exists() and not job.overwrite:
        return job.episode_index, "skipped"

    frames = extract_stacked_frames(job)
    states, actions = load_states_actions(job)

    trajectory = np.empty(len(job.frame_indices), dtype=object)
    for i, (image, state, action) in enumerate(zip(frames, states, actions)):
        trajectory[i] = {
            "image": image,
            "delta_action": action,
            "abs_action": action,
            "init_ee_pose": state,
            "instruction": TASK,
            "task": TASK,
            "episode_index": np.int64(job.episode_index),
            "source_frame_index": np.int64(job.frame_indices[i]),
            "source_action_index": np.int64(job.action_indices[i]),
        }

    tmp_path = out_path.with_suffix(".tmp.npy")
    np.save(tmp_path, trajectory, allow_pickle=True)
    os.replace(tmp_path, out_path)
    return job.episode_index, "written"


def build_jobs(args: argparse.Namespace) -> list[EpisodeJob]:
    source_root = args.source_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    info_path = source_root / "meta" / "info.json"
    episodes_path = source_root / "meta" / "episodes.jsonl"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    chunk_size = int(info.get("chunks_size", 1000))

    frame_indices = parse_int_tuple(args.frame_indices)
    action_indices = parse_int_tuple(args.action_indices)
    if len(frame_indices) != 5:
        raise ValueError(f"Wan Tower expects 5 condition frame indices, got {frame_indices}")
    if len(action_indices) != len(frame_indices):
        raise ValueError("action-indices must have the same length as frame-indices")

    episodes = load_jsonl(episodes_path)
    if args.max_episodes is not None:
        episodes = episodes[: args.max_episodes]

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    return [
        EpisodeJob(
            episode_index=int(record["episode_index"]),
            length=int(record["length"]),
            source_root=source_root,
            output_dir=output_dir,
            frame_indices=frame_indices,
            action_indices=action_indices,
            view_height=args.view_height,
            width=args.width,
            padding_bottom=args.padding_bottom,
            chunk_size=chunk_size,
            overwrite=args.overwrite,
            ffmpeg_exe=ffmpeg_exe,
        )
        for record in episodes
    ]


def run_jobs(jobs: Iterable[EpisodeJob], workers: int) -> dict[str, int]:
    jobs = list(jobs)
    counts = {"written": 0, "skipped": 0}
    total = len(jobs)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(convert_episode, job) for job in jobs]
        for done, future in enumerate(as_completed(futures), start=1):
            episode_index, status = future.result()
            counts[status] = counts.get(status, 0) + 1
            if done == 1 or done % 25 == 0 or done == total:
                print(
                    f"[{done:04d}/{total:04d}] episode {episode_index:06d}: {status} "
                    f"(written={counts.get('written', 0)}, skipped={counts.get('skipped', 0)})",
                    flush=True,
                )
    return counts


def main() -> None:
    args = parse_args()
    jobs = build_jobs(args)
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    counts = run_jobs(jobs, workers=args.workers)
    print(
        f"Done. output_dir={args.output_dir.resolve()} written={counts.get('written', 0)} "
        f"skipped={counts.get('skipped', 0)}"
    )


if __name__ == "__main__":
    main()
