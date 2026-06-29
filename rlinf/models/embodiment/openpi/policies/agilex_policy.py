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
"""Agilex (piper) dual-arm policy transforms for openpi.

Ported from OpenDriveLab/kai0 (``src/openpi/policies/agilex_policy.py``) so that
``pi05_piper`` SFT checkpoints can be loaded inside RLinf rollouts.

Key spec (matches kai0 piper SFT):
    * 14-dim state / action: (6 arm joints + 1 gripper) per arm, dual-arm.
      The gripper dims (index 6 and 13) are NOT delta'd; the joint dims are.
    * 3 cameras: top_head, hand_left, hand_right -> base_0_rgb,
      left_wrist_0_rgb, right_wrist_0_rgb.
    * action_horizon = 50, internal action_dim = 32 (state/action padded).

Difference vs kai0: wrist images are *optional* here. If only the top_head
image is present (e.g. when driven by the DreamDojo world model, which only
generates one view), the wrist slots are filled with zeros and their
``image_mask`` is set to False so PI05's masking handles them. The policy
will degrade vs the SFT distribution but will not crash.
"""
import dataclasses
from typing import ClassVar

import einops
import numpy as np
import torch
from openpi import transforms
from openpi.models import model as _model


def make_agilex_example() -> dict:
    """Creates a random input example for the Agilex/piper policy."""
    return {
        "images": {
            "top_head": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
            "hand_left": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
            "hand_right": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        },
        "state": np.random.rand(14),
        "prompt": "insert the battery into the mouse",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    image = np.squeeze(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class AgilexInputs(transforms.DataTransformFn):
    """Inputs for the Agilex/piper policy."""

    # Model action dimension; state and actions get padded to this.
    action_dim: int

    model_type: _model.ModelType = _model.ModelType.PI0

    # Strict kai0-style camera names -> openpi names.
    _CAM_RENAME: ClassVar[dict[str, str]] = {
        "top_head": "base_0_rgb",
        "hand_left": "left_wrist_0_rgb",
        "hand_right": "right_wrist_0_rgb",
    }
    REQUIRED_CAMERA: ClassVar[str] = "top_head"

    # If True, zero out the state (kai0's `mask_state` flag).
    mask_state: bool = False

    def __call__(self, data: dict) -> dict:
        # RL/inference path: ``openpi_action_model.obs_processor`` produces
        # ``observation/image`` / ``observation/wrist_image`` / ``observation/state``
        # keys. Rebuild the training-style ``data['images']`` / ``data['state']``
        # dict so the rest of __call__ stays format-agnostic.
        # (mirrors aloha_policy._decode_aloha.)
        if "observation/state" in data:
            base_img = _parse_image(data["observation/image"])
            images = {self.REQUIRED_CAMERA: base_img}
            if data.get("observation/wrist_image") is not None:
                wrist = np.asarray(data["observation/wrist_image"])
                # Dual wrist shape: [2, H, W, 3].
                if wrist.ndim == 4 and wrist.shape[0] == 2:
                    images["hand_left"] = wrist[0]
                    images["hand_right"] = wrist[1]
                else:
                    # Single wrist tensor goes to the left arm slot.
                    images["hand_left"] = wrist
            data = {**data, "images": images, "state": data["observation/state"]}

        in_images = data.get("images", {})
        if self.REQUIRED_CAMERA not in in_images:
            raise ValueError(
                f"AgilexInputs requires the '{self.REQUIRED_CAMERA}' camera; "
                f"got keys {list(in_images)}"
            )

        # State: keep first 14 dims (piper dual-arm 6+1+6+1), pad to model action_dim.
        state = data["state"]
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
        elif not isinstance(state, torch.Tensor):
            state = torch.as_tensor(state).float()
        state = state[..., :14]
        state = transforms.pad_to_dim(state, self.action_dim)
        state = state.squeeze()
        # Clamp NaN/out-of-range to 0 (matches kai0 robustness).
        if isinstance(state, torch.Tensor):
            s_np = state.numpy()
        else:
            s_np = np.asarray(state)
        s_np = np.where(s_np > np.pi, 0, s_np)
        s_np = np.where(s_np < -np.pi, 0, s_np)
        state = s_np

        # Cameras: fill present cameras; pad missing wrists with zeros + mask=False.
        images = {}
        image_masks = {}
        # Always populate top_head from the required camera.
        top_img = _parse_image(in_images[self.REQUIRED_CAMERA])
        base_name = self._CAM_RENAME[self.REQUIRED_CAMERA]

        if self.model_type in (_model.ModelType.PI0, _model.ModelType.PI05):
            for src_name, dst_name in self._CAM_RENAME.items():
                if src_name in in_images:
                    img = _parse_image(in_images[src_name])
                    images[dst_name] = img
                    image_masks[dst_name] = np.True_
                else:
                    images[dst_name] = np.zeros_like(top_img)
                    image_masks[dst_name] = np.False_
        elif self.model_type == _model.ModelType.PI0_FAST:
            # PI0-FAST does not mask images; use zeros for missing.
            for src_name, dst_name in self._CAM_RENAME.items():
                if src_name in in_images:
                    images[dst_name] = _parse_image(in_images[src_name])
                else:
                    images[dst_name] = np.zeros_like(top_img)
                image_masks[dst_name] = np.True_
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        masked_state = np.zeros_like(state) if self.mask_state else state
        inputs = {
            "state": masked_state,
            "image": images,
            "image_mask": image_masks,
        }

        if "actions" in data:
            actions = transforms.pad_to_dim(data["actions"], self.action_dim)
            actions = np.where(actions > np.pi, 0, actions)
            actions = np.where(actions < -np.pi, 0, actions)
            inputs["actions"] = actions.squeeze()

        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class AgilexOutputs(transforms.DataTransformFn):
    """Outputs for the Agilex/piper policy: slice the 32D action back to 14D."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :14])}
