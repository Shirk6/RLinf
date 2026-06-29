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

"""DreamDojo (Cosmos-Predict2 action-conditioned world model) environment.

DreamDojo is NVIDIA's action-conditioned video world model built on
``cosmos_predict2`` (NOT diffsynth). This env wraps the piper post-trained
checkpoint and exposes it through RLinf's :class:`BaseWorldEnv` contract so it
can be driven by an RL policy exactly like ``WanEnv``.

Key piper specifics (see DreamDojo ``configs/2b_1440_640_piper.yaml`` and
``groot_dreams/data/dataset.py``):

* The model consumes a 384-wide action vector per frame; piper's real action
  lives in slots ``[169:183]`` (14 dims = dual-arm joint-position deltas). All
  other slots are zero.
* ``lam_video`` (latent-action embeddings) only modulates slots ``[-32:]`` in
  the *training* forward; for piper those slots are zero, so ``lam_video`` has
  no effect and we pass ``None`` during rollout.
* Generation is image2world autoregressive: condition on the last frame
  (``num_latent_conditional_frames=1``), predict ``chunk`` future frames.

Reward is intentionally left as a zero stub here -- a dedicated reward model is
being developed separately. ``terminations`` are therefore always False and
episodes end only on truncation (``max_episode_steps``).
"""

import io
import os
import sys
from contextlib import nullcontext
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from rlinf.data.datasets.world_model import NpyTrajectoryDatasetWrapper
from rlinf.envs.utils import recursive_to_device
from rlinf.envs.world_model.base_world_env import BaseWorldEnv

__all__ = ["DreamDojoEnv"]


_COSMOS_RESOLVER_PATCHED = False


def _patch_omegaconf_resolvers():
    """Make cosmos's OmegaConf resolver registrations idempotent.

    cosmos_predict2 registers OmegaConf resolvers (``add``/``subtract``/...) at
    import time without ``replace=True``. RLinf already registers a ``subtract``
    resolver, so importing cosmos after rlinf raises ``ValueError: resolver
    'subtract' is already registered``. Patch ``register_new_resolver`` to default
    to ``replace=True`` so cosmos's (semantically compatible, variadic) versions
    win without error. Idempotent.
    """
    global _COSMOS_RESOLVER_PATCHED
    if _COSMOS_RESOLVER_PATCHED:
        return
    from omegaconf import OmegaConf

    _orig = OmegaConf.register_new_resolver

    def _patched(name, resolver, *args, **kwargs):
        kwargs.setdefault("replace", True)
        return _orig(name, resolver, *args, **kwargs)

    OmegaConf.register_new_resolver = _patched
    _COSMOS_RESOLVER_PATCHED = True


class DreamDojoEnv(BaseWorldEnv):
    """RLinf world-model env backed by the DreamDojo piper world model."""

    def __init__(
        self,
        cfg,
        num_envs,
        seed_offset,
        total_num_processes,
        record_metrics=True,
        worker_info=None,
    ):
        super().__init__(
            cfg, num_envs, seed_offset, total_num_processes, worker_info, record_metrics
        )

        # Reset-state management (mirrors WanEnv).
        self.use_fixed_reset_state_ids = cfg.get("use_fixed_reset_state_ids", True)
        self.group_size = cfg.get("group_size", 1)
        self.num_group = self.num_envs // self.group_size

        self._generator = torch.Generator()
        self._generator.manual_seed(self.seed)
        self.update_reset_state_ids()

        # DreamDojo / cosmos generation hyperparameters.
        self.chunk = cfg.chunk  # frames predicted per chunk_step (piper: 12)
        self.num_inference_steps = cfg.get("num_inference_steps", 35)
        self.guidance = cfg.get("guidance", 0)
        self.gen_height = cfg.get("height", 1440)
        self.gen_width = cfg.get("width", 640)
        self.num_latent_conditional_frames = cfg.get(
            "num_latent_conditional_frames", 1
        )
        self.seed_base = self.seed

        # Action layout inside the 384-wide cosmos action vector.
        self.model_action_dim = cfg.get("model_action_dim", 384)
        action_slot = cfg.get("action_slot", [169, 183])
        self.action_slot_start, self.action_slot_end = (
            int(action_slot[0]),
            int(action_slot[1]),
        )
        self.piper_action_dim = self.action_slot_end - self.action_slot_start  # 14

        # Resolution of the image returned to the policy.
        self.image_size = tuple(cfg.image_size)

        # Build the (persistent) cosmos inference pipeline.
        self.pipe = self._build_pipeline()
        self._negative_prompt = self._default_negative_prompt()

        # Per-env most-recent RGB frame, uint8 [num_envs, H, W, 3] in [0, 255].
        self.current_obs = None
        self.task_descriptions = [""] * self.num_envs
        self.init_ee_poses = [None] * self.num_envs

        self._is_offloaded = False

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    def _build_dataset(self, cfg):
        # Reuse the same npy initial-frame format as WanEnv. Use the piper
        # converter (convert_piper_to_initial_npy.py) to export LeRobot first
        # frames -> npy. Frames are kept at native resolution; the conditioning
        # frame is resized to (height, width) at rollout time.
        return NpyTrajectoryDatasetWrapper(
            cfg.initial_image_path,
            enable_kir=self.enable_kir,
        )

    def _ensure_dreamdojo_on_path(self):
        """Put the DreamDojo repo on sys.path so cosmos_predict2 is importable."""
        repo_path = self.cfg.get("dreamdojo_repo_path", None)
        if repo_path is not None and repo_path not in sys.path:
            sys.path.insert(0, str(repo_path))
        _patch_omegaconf_resolvers()

    def _build_pipeline(self):
        self._ensure_dreamdojo_on_path()
        from cosmos_predict2._src.predict2.inference.video2world import (
            Video2WorldInference,
        )

        pipe = Video2WorldInference(
            experiment_name=self.cfg.experiment,
            ckpt_path=self.cfg.dreamdojo_ckpt_path,
            s3_credential_path="",
            context_parallel_size=1,
            config_file=self.cfg.config_file,
        )
        return pipe

    def _default_negative_prompt(self):
        self._ensure_dreamdojo_on_path()
        try:
            from cosmos_predict2.config import DEFAULT_NEGATIVE_PROMPT

            return DEFAULT_NEGATIVE_PROMPT
        except Exception:
            return ""

    # ------------------------------------------------------------------
    # Reset-state sampling (same scheme as WanEnv)
    # ------------------------------------------------------------------
    def update_reset_state_ids(self):
        total_num_episodes = len(self.dataset)
        reset_state_ids = torch.randint(
            low=0,
            high=total_num_episodes,
            size=(self.num_group,),
            generator=self._generator,
        )
        self.reset_state_ids = reset_state_ids.repeat_interleave(
            repeats=self.group_size
        ).to(self.device)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    @torch.no_grad()
    def reset(
        self,
        *,
        seed: Optional[Union[int, list[int]]] = None,
        options: Optional[dict] = None,
        episode_indices: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ):
        self.onload()
        self.elapsed_steps = 0

        if self.is_start:
            if self.use_fixed_reset_state_ids:
                episode_indices = self.reset_state_ids
            self._is_start = False

        num_envs = self.num_envs
        if len(self.dataset) < num_envs:
            raise ValueError(
                f"Not enough episodes in dataset. Found {len(self.dataset)}, need {num_envs}"
            )

        if episode_indices is None:
            if seed is not None:
                np.random.seed(seed[0] if isinstance(seed, list) else seed)
            episode_indices = np.random.choice(
                len(self.dataset), size=num_envs, replace=False
            )
        elif isinstance(episode_indices, torch.Tensor):
            episode_indices = episode_indices.cpu().numpy()

        frames = []  # list of uint8 [H, W, 3]
        task_descriptions = []
        init_ee_poses = []

        for episode_idx in episode_indices:
            episode_data = self.dataset[int(episode_idx)]
            if len(episode_data["start_items"]) == 0:
                raise ValueError(f"Empty start_items for episode {episode_idx}")
            first_frame = episode_data["start_items"][0]
            task_descriptions.append(str(episode_data.get("task", "")))

            if "image" not in first_frame:
                raise ValueError(f"No 'image' key in frame for episode {episode_idx}")
            # NpyTrajectoryDatasetWrapper returns CHW float in [0, 1].
            img = first_frame["image"]  # [3, H, W]
            # Resize conditioning frame to native generation resolution.
            if tuple(img.shape[1:]) != (self.gen_height, self.gen_width):
                img = F.interpolate(
                    img.unsqueeze(0),
                    size=(self.gen_height, self.gen_width),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
            img = (img.clamp(0, 1) * 255.0).to(torch.uint8)  # [3, H, W]
            img = img.permute(1, 2, 0).contiguous()  # [H, W, 3]
            frames.append(img)

            init_ee_poses.append(
                first_frame["observation.state"].numpy()
                if "observation.state" in first_frame
                else None
            )

        # [num_envs, H, W, 3] uint8 on device.
        self.current_obs = torch.stack(frames, dim=0).to(self.device)
        self.task_descriptions = task_descriptions
        self.init_ee_poses = init_ee_poses

        self._reset_metrics()
        return self._wrap_obs(), {}

    # ------------------------------------------------------------------
    # Stepping
    # ------------------------------------------------------------------
    @torch.no_grad()
    def step(self, actions=None, auto_reset=True):
        raise NotImplementedError(
            "step in DreamDojoEnv is not implemented, use chunk_step instead"
        )

    def _build_model_action(self, env_action):
        """Scatter a piper [chunk, 14] action into a [chunk, 384] cosmos vector."""
        if isinstance(env_action, np.ndarray):
            env_action = torch.from_numpy(env_action)
        env_action = env_action.float()
        chunk = env_action.shape[0]
        model_action = torch.zeros(
            chunk, self.model_action_dim, dtype=torch.float32
        )
        model_action[:, self.action_slot_start : self.action_slot_end] = env_action[
            :, : self.piper_action_dim
        ]
        return model_action

    @torch.no_grad()
    def _infer_next_chunk_frames(self, actions):
        """Autoregressively generate the next chunk for each env."""
        assert actions.shape[0] == self.num_envs, (
            f"Actions shape {actions.shape} does not match num_envs {self.num_envs}"
        )
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu()

        num_video_frames = self.chunk + 1
        new_frames = []  # last frame per env, uint8 [H, W, 3]

        for env_idx in range(self.num_envs):
            # Build conditioning video: first frame = current obs, rest zeros.
            last_frame = self.current_obs[env_idx]  # [H, W, 3] uint8
            img = last_frame.permute(2, 0, 1).unsqueeze(0).float()  # [1, 3, H, W]
            vid_input = torch.cat(
                [img, torch.zeros_like(img).repeat(num_video_frames - 1, 1, 1, 1)],
                dim=0,
            )  # [T, 3, H, W]
            vid_input = (
                vid_input.to(torch.uint8).unsqueeze(0).permute(0, 2, 1, 3, 4)
            )  # [1, 3, T, H, W]

            model_action = self._build_model_action(actions[env_idx])

            video = self.pipe.generate_vid2world(
                prompt="",
                input_path=vid_input,
                action=model_action,
                guidance=self.guidance,
                num_video_frames=num_video_frames,
                num_latent_conditional_frames=self.num_latent_conditional_frames,
                resolution="none",
                seed=self.seed_base + self.elapsed_steps + env_idx,
                negative_prompt=self._negative_prompt,
                num_steps=self.num_inference_steps,
                lam_video=None,
            )
            # video: [1, 3, T, H, W] in [-1, 1]; take the last predicted frame.
            video = (video + 1.0) / 2.0
            last = (
                torch.clamp(video[0, :, -1], 0, 1) * 255.0
            ).to(torch.uint8)  # [3, H, W]
            new_frames.append(last.permute(1, 2, 0).contiguous())  # [H, W, 3]

            del video
            self._clear_accelerator_cache()

        self.current_obs = torch.stack(new_frames, dim=0).to(self.device)

    def _infer_next_chunk_rewards(self):
        """Reward stub: a dedicated reward model is developed separately.

        Returns zeros of shape [num_envs, chunk]. Replace this once the piper
        reward model is available (cf. WanEnv._infer_next_chunk_rewards).
        """
        return torch.zeros(
            self.num_envs, self.chunk, dtype=torch.float32, device=self.device
        )

    def _calc_step_reward(self, chunk_rewards):
        if self.use_rel_reward:
            reward_diffs = torch.zeros(
                (self.num_envs, self.chunk), dtype=torch.float32, device=self.device
            )
            for i in range(self.chunk):
                reward_diffs[:, i] = (
                    self.cfg.get("reward_coef", 1.0) * chunk_rewards[:, i]
                    - self.prev_step_reward
                )
                self.prev_step_reward = (
                    self.cfg.get("reward_coef", 1.0) * chunk_rewards[:, i]
                )
            return reward_diffs
        return chunk_rewards

    def _wrap_obs(self):
        """Return obs dict aligned with libero_env / WanEnv format."""
        full_image = self.current_obs  # [num_envs, H, W, 3] uint8
        if tuple(full_image.shape[1:3]) != self.image_size:
            x = full_image.permute(0, 3, 1, 2).float()  # [N, 3, H, W]
            x = F.interpolate(
                x, size=self.image_size, mode="bilinear", align_corners=False
            )
            full_image = x.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)

        states = torch.zeros(
            (self.num_envs, 16), device=self.device, dtype=torch.float32
        )
        return {
            "main_images": full_image,
            "wrist_images": None,
            "states": states,
            "task_descriptions": self.task_descriptions,
        }

    def _handle_auto_reset(self, dones, extracted_obs, infos):
        final_obs = extracted_obs
        final_info = infos
        extracted_obs, infos = self.reset()
        infos["final_observation"] = final_obs
        infos["final_info"] = final_info
        infos["_final_info"] = dones
        infos["_final_observation"] = dones
        infos["_elapsed_steps"] = dones
        return extracted_obs, infos

    @torch.no_grad()
    def chunk_step(self, policy_output_action):
        """Advance one chunk. ``policy_output_action``: [num_envs, chunk, 14]."""
        self.onload()
        autocast_context = (
            torch.amp.autocast(device_type=self.device.type, dtype=torch.bfloat16)
            if self.device.type != "cpu"
            else nullcontext()
        )
        with autocast_context:
            self._infer_next_chunk_frames(policy_output_action)

        self.elapsed_steps += self.chunk
        extracted_obs = self._wrap_obs()

        chunk_rewards = self._infer_next_chunk_rewards()
        chunk_rewards_tensors = self._calc_step_reward(chunk_rewards)

        # Reward model not available -> no success detection; terminate only on
        # truncation (max_episode_steps).
        raw_chunk_terminations = torch.zeros(
            self.num_envs, self.chunk, dtype=torch.bool, device=self.device
        )
        raw_chunk_truncations = torch.zeros(
            self.num_envs, self.chunk, dtype=torch.bool, device=self.device
        )
        truncations = torch.tensor(
            self.elapsed_steps >= self.cfg.max_episode_steps, device=self.device
        )
        if truncations.any():
            raw_chunk_truncations[:, -1] = truncations

        past_terminations = raw_chunk_terminations.any(dim=1)
        past_truncations = raw_chunk_truncations.any(dim=1)
        past_dones = torch.logical_or(past_terminations, past_truncations)

        if past_dones.any() and self.auto_reset:
            extracted_obs, infos = self._handle_auto_reset(past_dones, extracted_obs, {})
        else:
            infos = {}

        infos = self._record_metrics(
            chunk_rewards_tensors.sum(dim=1), past_terminations, infos
        )

        chunk_terminations = torch.zeros_like(raw_chunk_terminations)
        chunk_terminations[:, -1] = past_terminations
        chunk_truncations = torch.zeros_like(raw_chunk_truncations)
        chunk_truncations[:, -1] = past_truncations

        return (
            [extracted_obs],
            chunk_rewards_tensors,
            chunk_terminations,
            chunk_truncations,
            [infos],
        )

    # ------------------------------------------------------------------
    # Offload / state (for RLinf memory management)
    # ------------------------------------------------------------------
    def _move_pipe(self, device):
        for attr in ("dit", "net", "model", "vae", "tokenizer", "text_encoder"):
            module = getattr(self.pipe, attr, None)
            if module is not None and hasattr(module, "to"):
                try:
                    setattr(self.pipe, attr, module.to(device))
                except Exception:
                    pass

    def offload(self):
        if self._is_offloaded:
            return
        self._move_pipe("cpu")
        if self.current_obs is not None:
            self.current_obs = recursive_to_device(self.current_obs, "cpu")
        self.prev_step_reward = self.prev_step_reward.cpu()
        self.reset_state_ids = self.reset_state_ids.cpu()
        if self.record_metrics:
            self.success_once = self.success_once.cpu()
            self.returns = self.returns.cpu()
        self._clear_accelerator_cache()
        self._is_offloaded = True

    def onload(self):
        if not self._is_offloaded:
            return
        self._move_pipe(self.device)
        if self.current_obs is not None:
            self.current_obs = recursive_to_device(self.current_obs, self.device)
        self.prev_step_reward = self.prev_step_reward.to(self.device)
        self.reset_state_ids = self.reset_state_ids.to(self.device)
        if self.record_metrics:
            self.success_once = self.success_once.to(self.device)
            self.returns = self.returns.to(self.device)
        self._is_offloaded = False

    def get_state(self) -> bytes:
        env_state = {
            "current_obs": recursive_to_device(self.current_obs, "cpu")
            if self.current_obs is not None
            else None,
            "task_descriptions": self.task_descriptions,
            "init_ee_poses": self.init_ee_poses,
            "elapsed_steps": self.elapsed_steps,
            "prev_step_reward": self.prev_step_reward.cpu(),
            "_is_start": self._is_start,
            "reset_state_ids": self.reset_state_ids.cpu(),
            "generator_state": self._generator.get_state(),
        }
        if self.record_metrics:
            env_state.update(
                {
                    "success_once": self.success_once.cpu(),
                    "returns": self.returns.cpu(),
                }
            )
        buffer = io.BytesIO()
        torch.save(env_state, buffer)
        return buffer.getvalue()


if __name__ == "__main__":
    from pathlib import Path

    from hydra import compose
    from hydra.core.global_hydra import GlobalHydra
    from hydra.initialize import initialize_config_dir

    os.environ.setdefault("EMBODIED_PATH", "examples/embodiment")
    repo_root = Path(__file__).resolve().parents[3]
    GlobalHydra.instance().clear()
    config_dir = Path(
        os.environ.get("EMBODIED_CONFIG_DIR", repo_root / "examples/embodiment/config")
    ).resolve()
    config_name = os.environ.get("DREAMDOJO_CONFIG", "dreamdojo_piper_grpo")

    print(f"Loading config: {config_name} from {config_dir}")
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
        cfg_ = compose(config_name=config_name)
        cfg = cfg_["env"]["train"]

    env = DreamDojoEnv(cfg, cfg.total_num_envs, seed_offset=0, total_num_processes=1)
    obs, info = env.reset()
    print("After reset, obs keys:", list(obs.keys()))
    print("main_images:", obs["main_images"].shape, obs["main_images"].dtype)

    zeros_actions = np.zeros((cfg.total_num_envs, cfg.chunk, 14), dtype=np.float32)
    o, r, te, tr, infos = env.chunk_step(zeros_actions)
    print("After chunk_step:", o[0]["main_images"].shape, "reward", r.shape)
