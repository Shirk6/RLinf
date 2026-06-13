#!/usr/bin/env python3
"""Build Wan Tower initial-state trajectories from LeRobot expert data.

The Wan Tower environment reads one ``.npy`` trajectory per reset state through
``NpyTrajectoryDatasetWrapper``.  This script converts the Challenge phase-1
Tower of Hanoi expert dataset into that format and writes three initial-state
sets:

* ``video_start`` starts from source frame 0;
* ``stage1`` starts from the manually labeled stage_1 frame;
* ``stage2`` starts from the manually labeled stage_2 frame;
* each trajectory keeps the whole episode's first frame and contains 5 frames
  total. ``video_start`` uses [0, 6, 12, 18, 24]; stage trajectories use
  [0, start, start + 6, start + 12, start + 18];
* actions are sampled at the same source indices as frames;
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
DEFAULT_LABELS_PATH = Path(
    "Challenge-phase1-dataset/tower-of-hanoi-game/manual_stage_labels/labels_merged.json"
)
INITIAL_STATE_SPECS = (
    ("video_start", None),
    ("stage1", "stage_1"),
    ("stage2", "stage_2"),
)
NUM_CONDITION_FRAMES = 5
FRAME_STRIDE = 6
TASK = "tower-of-hanoi-game"


@dataclass(frozen=True)
class EpisodeJob:
    episode_index: int
    episode_id: str
    initial_state: str
    initial_frame_index: int
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
        help=(
            "Base output directory. The script writes video_start/, stage1/, "
            "and stage2/ subdirectories under this path."
        ),
    )
    parser.add_argument(
        "--labels-path",
        type=Path,
        default=DEFAULT_LABELS_PATH,
        help="Merged manual stage labels JSON used for stage1/stage2 starts.",
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


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_stage_labels(labels_path: Path, source_name: str) -> list[dict]:
    data = json.loads(labels_path.read_text(encoding="utf-8-sig"))
    raw_labels = data.get("labels", data)
    if not isinstance(raw_labels, dict):
        raise ValueError(f"{labels_path} does not contain a labels object")

    records = []
    for episode_id, record in raw_labels.items():
        if not isinstance(record, dict):
            continue
        if str(record.get("source")) != source_name:
            continue
        stages = record.get("stages") or {}
        if stages.get("stage_1") is None or stages.get("stage_2") is None:
            continue
        labeled_episode_id = str(record.get("episode_id") or episode_id)
        records.append({**record, "episode_id": labeled_episode_id})

    records.sort(key=lambda item: int(item["episode_index"]))
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


def sampled_indices(start_frame: int) -> tuple[int, ...]:
    if start_frame == 0:
        return tuple(i * FRAME_STRIDE for i in range(NUM_CONDITION_FRAMES))
    return (0,) + tuple(
        start_frame + i * FRAME_STRIDE for i in range(NUM_CONDITION_FRAMES - 1)
    )


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
            "episode_id": job.episode_id,
            "initial_state": job.initial_state,
            "initial_frame_index": np.int64(job.initial_frame_index),
            "source_frame_index": np.int64(job.frame_indices[i]),
            "source_action_index": np.int64(job.action_indices[i]),
        }

    tmp_path = out_path.with_suffix(".tmp.npy")
    np.save(tmp_path, trajectory, allow_pickle=True)
    os.replace(tmp_path, out_path)
    return job.episode_index, "written"


def build_jobs(args: argparse.Namespace) -> list[EpisodeJob]:
    source_root = args.source_root.resolve()
    output_base_dir = args.output_dir.resolve()
    output_base_dir.mkdir(parents=True, exist_ok=True)
    labels_path = args.labels_path.resolve()

    info_path = source_root / "meta" / "info.json"
    episodes_path = source_root / "meta" / "episodes.jsonl"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    chunk_size = int(info.get("chunks_size", 1000))

    episodes_by_index = {
        int(record["episode_index"]): record for record in load_jsonl(episodes_path)
    }
    label_records = load_stage_labels(labels_path, source_root.name)
    if args.max_episodes is not None:
        label_records = label_records[: args.max_episodes]

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    jobs = []
    skipped_missing_episode = 0
    skipped_too_short = 0

    for label_record in label_records:
        episode_index = int(label_record["episode_index"])
        episode_record = episodes_by_index.get(episode_index)
        if episode_record is None:
            skipped_missing_episode += 1
            continue

        length = int(episode_record["length"])
        stages = label_record.get("stages") or {}
        for initial_state, stage_key in INITIAL_STATE_SPECS:
            if stage_key is None:
                start_frame = 0
            else:
                stage_value = stages.get(stage_key)
                if stage_value is None:
                    continue
                start_frame = int(stage_value["frame"])

            frame_indices = sampled_indices(start_frame)
            if frame_indices[-1] >= length:
                skipped_too_short += 1
                print(
                    f"Warning: skip episode {episode_index:06d} {initial_state}; "
                    f"need frame {frame_indices[-1]}, length is {length}",
                    flush=True,
                )
                continue

            output_dir = output_base_dir / initial_state
            output_dir.mkdir(parents=True, exist_ok=True)
            jobs.append(
                EpisodeJob(
                    episode_index=episode_index,
                    episode_id=str(label_record["episode_id"]),
                    initial_state=initial_state,
                    initial_frame_index=start_frame,
                    length=length,
                    source_root=source_root,
                    output_dir=output_dir,
                    frame_indices=frame_indices,
                    action_indices=frame_indices,
                    view_height=args.view_height,
                    width=args.width,
                    padding_bottom=args.padding_bottom,
                    chunk_size=chunk_size,
                    overwrite=args.overwrite,
                    ffmpeg_exe=ffmpeg_exe,
                )
            )

    if not jobs:
        raise ValueError(
            f"No jobs built from labels_path={labels_path} and source={source_root.name}"
        )
    if skipped_missing_episode:
        print(
            f"Warning: skipped {skipped_missing_episode} labels with no matching episode metadata",
            flush=True,
        )
    if skipped_too_short:
        print(
            f"Warning: skipped {skipped_too_short} initial states that were too close to episode end",
            flush=True,
        )
    return jobs


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
