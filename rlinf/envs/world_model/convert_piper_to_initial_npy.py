# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Export piper LeRobot episodes into per-episode .npy initial-frame files.

DreamDojoEnv reuses RLinf's :class:`NpyTrajectoryDatasetWrapper` for resets,
which expects a directory of ``*.npy`` files, one per episode, each holding an
object array of per-frame dicts with keys ``image`` (HWC uint8), ``delta_action``
and ``instruction``. This script reads the piper LeRobot dataset through
DreamDojo's ``MultiVideoActionDataset`` and writes those npy files.

Run inside the DreamDojo venv (so ``groot_dreams`` imports), e.g.::

    cd /mnt/afs-h200/yuyangcheng/data/Shirk6_DreamDojo_piper
    source .venv/bin/activate
    PYTHONPATH=. python /path/to/RLinf/rlinf/envs/world_model/convert_piper_to_initial_npy.py \
        --dataset-path datasets/piper_insert_mouse_battery_lerobot \
        --out-dir /mnt/afs-h200/yuyangcheng/data/piper_initial_frames \
        --num-episodes 64

The ``--out-dir`` value is what you set as ``env.train.initial_image_path`` in
``examples/embodiment/config/env/dreamdojo_piper.yaml``.
"""

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--num-episodes", type=int, default=64)
    parser.add_argument("--frames-per-file", type=int, default=5)
    parser.add_argument("--num-frames", type=int, default=13)
    parser.add_argument("--height", type=int, default=1440)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--video-key", default="video.cam_vertical")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--data-split", default="train")
    parser.add_argument("--instruction", default="insert the battery")
    parser.add_argument("--piper-action-dim", type=int, default=14)
    args = parser.parse_args()

    from groot_dreams.dataloader import MultiVideoActionDataset

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = MultiVideoActionDataset(
        dataset_path=args.dataset_path,
        num_frames=args.num_frames,
        data_split=args.data_split,
        restrict_len=args.num_episodes,
        height=args.height,
        width=args.width,
        video_key=args.video_key,
        fps=args.fps,
    )

    n = min(args.num_episodes, len(dataset))
    print(f"Dataset has {len(dataset)} samples; exporting {n} episodes.")

    for idx in range(n):
        data = dataset[idx]
        video = data["video"]  # (C, T, H, W) uint8
        video = video.detach().cpu().numpy()
        c, t, h, w = video.shape
        k = min(args.frames_per_file, t)

        instruction = data.get("ai_caption") or args.instruction
        if isinstance(instruction, (list, tuple)):
            instruction = instruction[0] if instruction else args.instruction
        instruction = str(instruction) if instruction else args.instruction

        frames = []
        for f in range(k):
            img = np.ascontiguousarray(
                video[:, f].transpose(1, 2, 0)
            ).astype(np.uint8)  # (H, W, 3)
            frames.append(
                {
                    "image": img,
                    "delta_action": np.zeros(args.piper_action_dim, dtype=np.float32),
                    "instruction": instruction,
                }
            )

        arr = np.empty(len(frames), dtype=object)
        for i, fr in enumerate(frames):
            arr[i] = fr
        out_path = out_dir / f"episode_{idx:05d}.npy"
        np.save(out_path, arr, allow_pickle=True)

    print(f"Wrote {n} npy files to {out_dir}")


if __name__ == "__main__":
    main()
