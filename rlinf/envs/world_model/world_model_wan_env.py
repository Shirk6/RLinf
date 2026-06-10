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

import io
import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from diffsynth.models.reward_model import ResnetRewModel, TaskEmbedResnetRewModel
from diffsynth.pipelines.wan_video_new import ModelConfig, WanVideoPipeline
from omegaconf import OmegaConf
from PIL import Image

from rlinf.data.datasets.world_model import NpyTrajectoryDatasetWrapper
from rlinf.envs.utils import recursive_to_device
from rlinf.envs.world_model.base_world_env import BaseWorldEnv
from rlinf.models.embodiment.reward.resnet_reward_model import ResNetRewardModel

__all__ = ["WanEnv"]


class WanEnv(BaseWorldEnv):
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
        # Reset state management
        self.use_fixed_reset_state_ids = cfg.use_fixed_reset_state_ids
        self.group_size = cfg.group_size
        self.num_group = self.num_envs // self.group_size

        # Initialize reset state generator
        self._generator = torch.Generator()
        self._generator.manual_seed(self.seed)

        # Update reset state ids
        self.update_reset_state_ids()

        # Model hyperparameters
        self.num_inference_steps = cfg.num_inference_steps
        self.chunk = cfg.chunk
        self.action_dim = cfg.get("action_dim", 7)
        self.condition_frame_length = cfg.condition_frame_length
        self.num_frames = cfg.num_frames
        self.wan_predict_frames = cfg.get(
            "wan_predict_frames", self.num_frames - self.condition_frame_length
        )
        self.action_downsample_stride = cfg.get("action_downsample_stride", 1)
        assert self.num_frames == self.condition_frame_length + self.wan_predict_frames, (
            "num_frames must be equal to condition_frame_length + wan_predict_frames"
        )
        if self.wan_predict_frames <= 0:
            raise ValueError(
                f"wan_predict_frames must be positive, got {self.wan_predict_frames}"
            )
        if self.action_downsample_stride <= 0:
            raise ValueError(
                "action_downsample_stride must be positive, "
                f"got {self.action_downsample_stride}"
            )
        if self.action_downsample_stride * self.wan_predict_frames > self.chunk:
            raise ValueError(
                "action_downsample_stride * wan_predict_frames must fit within chunk: "
                f"{self.action_downsample_stride} * {self.wan_predict_frames} > {self.chunk}"
            )

        self.image_size = tuple(cfg.image_size)
        self.image_layout = cfg.get("image_layout", "single")
        self.view_height = cfg.get("view_height", 180)
        self.num_views = cfg.get("num_views", 3)
        self.padding_bottom = cfg.get("padding_bottom", 0)
        if self.image_layout == "vertical_3view":
            if self.image_size[0] % 3 != 0:
                raise ValueError(
                    f"vertical_3view requires image height divisible by 3, got {self.image_size}"
                )
        elif self.image_layout == "vertical_3view_padded_bottom":
            valid_height = self.view_height * self.num_views
            expected_height = valid_height + self.padding_bottom
            if self.image_size[0] != expected_height:
                raise ValueError(
                    "vertical_3view_padded_bottom requires image height "
                    f"{expected_height} (= {self.num_views} * {self.view_height} + "
                    f"{self.padding_bottom}), got {self.image_size}"
                )
        elif self.image_layout != "single":
            raise ValueError(f"Unknown Wan image_layout: {self.image_layout}")

        #
        self.retain_action = cfg.get("retain_action", True)  # Default True
        self.enable_kir = cfg.get("enable_kir", True)
        self.use_latent_condition_cache = cfg.get("use_latent_condition_cache", True)

        # load pipeline
        self.pipe = self._build_pipeline()

        # Load reward model if specified
        self.reward_model = self._load_reward_model().eval().to(self.device)

        # Initialize state
        # Will be a tensor [num_envs, 3, 1, T, h, w]
        self.current_obs = None
        self.condition_latents = None
        self.task_descriptions = [""] * self.num_envs
        self.init_ee_poses = [None] * self.num_envs
        self.state_proxy = torch.zeros(
            self.num_envs, self.action_dim, device=self.device, dtype=torch.float32
        )

        # Image queue for condition frames to generate video. The first frame
        # is the reset reference frame and remains fixed during autoregressive rollout.
        self.image_queue = [
            [None] * self.condition_frame_length for _ in range(self.num_envs)
        ]

        # Condition action to generate video,
        # keep length of condition_frame_length
        self.condition_action = torch.zeros(
            self.num_envs,
            self.condition_frame_length,
            self.action_dim,
        )

        self.reset_gripper_open = cfg.get("reset_gripper_open", True)
        self.is_libero_env = cfg.get("wm_env_type", "libero") == "libero"

        # If reset_gripper_open is True and the environment is Libero, set the gripper open action to -1
        if self.reset_gripper_open and self.is_libero_env:
            self.condition_action[:, :, -1] = -1

        self.trans_norm = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True
                ),
            ]
        )

        self._is_offloaded = False

    def _build_dataset(self, cfg):
        return NpyTrajectoryDatasetWrapper(
            cfg.initial_image_path, enable_kir=self.enable_kir
        )

    def _build_pipeline(self):
        pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cuda:0",
            model_configs=[
                # Paths are loaded from yaml
                ModelConfig(path=self.cfg.model_path, offload_device="cpu"),
                ModelConfig(path=self.cfg.VAE_path, offload_device="cpu"),
            ],
        )
        # pipe.enable_vram_management()
        pipe.dit.to(self.device)
        pipe.vae.to(self.device)
        return pipe

    def _load_reward_model(self):
        if self.cfg.reward_model.type == "ResnetRewModel":
            rew_model = ResnetRewModel(self.cfg.reward_model.from_pretrained)
        elif self.cfg.reward_model.type == "ResNetRewardModel":
            reward_model_cfg = OmegaConf.create(
                OmegaConf.to_container(self.cfg.reward_model, resolve=True)
            )
            reward_model_cfg.model_path = self.cfg.reward_model.from_pretrained
            rew_model = ResNetRewardModel(reward_model_cfg)
        elif self.cfg.reward_model.type == "TaskEmbedResnetRewModel":
            rew_model = TaskEmbedResnetRewModel(
                checkpoint_path=self.cfg.reward_model.from_pretrained,
                task_suite_name=self.cfg.task_suite_name,
            )
        else:
            raise ValueError(f"Unknown reward model type: {self.cfg.reward_model.type}")
        return rew_model

    def _init_metrics(self):
        self.success_once = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.returns = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32
        )

    def _reset_metrics(self, env_idx=None):
        if env_idx is not None:
            mask = torch.zeros(self.num_envs, dtype=bool, device=self.device)
            mask[env_idx] = True
            self.prev_step_reward[mask] = 0.0
            if self.record_metrics:
                self.success_once[mask] = False
                self.returns[mask] = 0
            # self._elapsed_steps = 0
        else:
            self.prev_step_reward[:] = 0
            if self.record_metrics:
                self.success_once[:] = False
                self.returns[:] = 0.0
            self._elapsed_steps = 0

    def _record_metrics(self, step_reward, terminations, infos):
        episode_info = {}
        self.returns += step_reward
        # Update success_once based on terminations
        if isinstance(terminations, torch.Tensor):
            self.success_once = self.success_once | terminations
        else:
            terminations_tensor = torch.tensor(
                terminations, device=self.device, dtype=torch.bool
            )
            self.success_once = self.success_once | terminations_tensor
        episode_info["success_once"] = self.success_once.clone()
        episode_info["return"] = self.returns.clone()
        episode_info["episode_len"] = torch.full(
            (self.num_envs,),
            self.elapsed_steps,
            dtype=torch.float32,
            device=self.device,
        )
        episode_info["reward"] = episode_info["return"] / episode_info["episode_len"]
        infos["episode"] = episode_info
        return infos

    def _calc_step_reward(self, chunk_rewards):
        """Calculate step reward"""
        reward_diffs = torch.zeros(
            (self.num_envs, self.chunk), dtype=torch.float32, device=self.device
        )
        for i in range(self.chunk):
            reward_diffs[:, i] = (
                self.cfg.reward_coef * chunk_rewards[:, i] - self.prev_step_reward
            )
            self.prev_step_reward = self.cfg.reward_coef * chunk_rewards[:, i]

        if self.use_rel_reward:
            return reward_diffs
        else:
            return chunk_rewards

    def _estimate_success_from_rewards(self, chunk_rewards):
        """
        Estimate success (terminations) based on reward values.
        Success is estimated when reward exceeds a threshold (default: 0.9).
        """
        # Get success threshold from config, default to 0.9
        success_threshold = getattr(self.cfg, "success_reward_threshold", 0.9)

        # Check if any reward in the chunk exceeds the threshold
        # chunk_rewards shape: [num_envs, chunk]
        max_reward_in_chunk = chunk_rewards.max(dim=1)[0]  # [num_envs]
        success_estimated = max_reward_in_chunk >= success_threshold

        return success_estimated.to(self.device)

    def _downsample_actions_for_wan(self, chunk_actions):
        action_indices = (
            torch.arange(
                self.wan_predict_frames,
                device=chunk_actions.device,
                dtype=torch.long,
            )
            * self.action_downsample_stride
            + (self.action_downsample_stride - 1)
        )
        return chunk_actions.index_select(dim=1, index=action_indices)

    def _expand_wan_rewards_to_chunk(self, wan_rewards):
        if wan_rewards.shape != (self.num_envs, self.wan_predict_frames):
            raise ValueError(
                "Unexpected Wan reward shape "
                f"{wan_rewards.shape}; expected {(self.num_envs, self.wan_predict_frames)}"
            )
        expanded = wan_rewards.repeat_interleave(self.action_downsample_stride, dim=1)
        if expanded.shape[1] < self.chunk:
            pad = expanded[:, -1:].expand(-1, self.chunk - expanded.shape[1])
            expanded = torch.cat([expanded, pad], dim=1)
        return expanded[:, : self.chunk]

    def update_reset_state_ids(self):
        """Updates the reset state IDs for environment initialization."""
        # Get total number of episodes available
        total_num_episodes = len(self.dataset)

        # Generate random reset state ids
        reset_state_ids = torch.randint(
            low=0,
            high=total_num_episodes,
            size=(self.num_group,),
            generator=self._generator,
        )

        # Repeat for each environment in the group
        self.reset_state_ids = reset_state_ids.repeat_interleave(
            repeats=self.group_size
        ).to(self.device)

    @torch.no_grad()
    def reset(
        self,
        *,
        seed: Optional[Union[int, list[int]]] = None,
        options: Optional[dict] = {},
        episode_indices: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ):
        self.onload()
        self.elapsed_steps = 0

        # Handle first reset with fixed reset state ids
        if self.is_start:
            if self.use_fixed_reset_state_ids:
                episode_indices = self.reset_state_ids
            self._is_start = False

        num_envs = self.num_envs
        if len(self.dataset) < num_envs:
            raise ValueError(
                f"Not enough episodes in dataset. Found {len(self.dataset)}, need {num_envs}"
            )

        # If episode_indices not provided, randomly select
        if episode_indices is None:
            # Set random seed if provided
            if seed is not None:
                if isinstance(seed, list):
                    np.random.seed(seed[0])
                else:
                    np.random.seed(seed)

            # Randomly select episode indices
            episode_indices = np.random.choice(
                len(self.dataset), size=num_envs, replace=False
            )
        else:
            # Convert to numpy if tensor
            if isinstance(episode_indices, torch.Tensor):
                episode_indices = episode_indices.cpu().numpy()

        # Load first frame from each selected episode
        img_tensors = []
        task_descriptions = []
        init_ee_poses = []
        condition_actions = []

        for env_idx, episode_idx in enumerate(episode_indices):
            # Get episode data from dataset wrapper
            episode_data = self.dataset[episode_idx]

            # Get first frame from start_items
            if len(episode_data["start_items"]) == 0:
                raise ValueError(f"Empty start_items for episode {episode_idx}")

            first_frame = episode_data["start_items"][0]

            # Get task description
            task_desc = episode_data.get("task", "")
            task_descriptions.append(str(task_desc))

            # Get image from frame
            if "image" not in first_frame:
                raise ValueError(f"No 'image' key in frame for episode {episode_idx}")

            img_tensor = first_frame[
                "image"
            ]  # Shape: [3, H, W], dtype: float, range: [0, 1]
            # [3, 256, 256], float32, [0,1]
            # Wan requires images in PIL format

            # Get init_ee_pose if available
            if "observation.state" in first_frame:
                init_ee_poses.append(first_frame["observation.state"].numpy())
            else:
                init_ee_poses.append(None)

            # Resize if needed
            if img_tensor.shape[1:] != self.image_size:
                img_tensor = img_tensor.unsqueeze(0)  # [1, 3, H, W]
                img_tensor = F.interpolate(
                    img_tensor,
                    size=self.image_size,
                    mode="bilinear",
                    align_corners=False,
                )
                img_tensor = img_tensor.squeeze(0)  # [3, H, W]

            # Normalize to [-1, 1]
            img_tensor = self.trans_norm(img_tensor)

            # Repeat to fill condition frames: [3, H, W] -> [3, condition_frame_length, H, W]
            env_img_tensor = img_tensor.unsqueeze(1).repeat(
                1, self.condition_frame_length, 1, 1
            )  # [3, condition_frame_length, H, W]

            env_condition_action = np.zeros(
                (self.condition_frame_length, self.action_dim), dtype=np.float32
            )

            if self.reset_gripper_open and self.is_libero_env:
                env_condition_action[:, -1] = -1

            # KIR trick: use the last four frames as condition frames, while
            # keeping the reference frame unchanged as the first frame.
            target_items = episode_data.get("target_items", [])

            # first condition frame is the reference frame,
            # so the length of target_items should be condition_frame_length - 1
            if len(target_items) == self.condition_frame_length - 1:
                for target_idx, target_frame in enumerate(target_items):
                    if "image" not in target_frame or "action" not in target_frame:
                        raise ValueError(
                            f"No 'image' or 'action' key in target frame for episode {episode_idx}"
                        )
                    target_img = target_frame["image"]
                    if target_img.shape[1:] != self.image_size:
                        target_img = target_img.unsqueeze(0)  # [1, 3, H, W]
                        target_img = F.interpolate(
                            target_img,
                            size=self.image_size,
                            mode="bilinear",
                            align_corners=False,
                        )
                        target_img = target_img.squeeze(0)  # [3, H, W]
                    target_img = self.trans_norm(target_img)

                    # keep first frame as reference frame, update the rest
                    env_img_tensor[:, target_idx + 1] = target_img
                    env_condition_action[target_idx + 1] = self._fit_action_dim(
                        target_frame["action"]
                    )

            img_tensors.append(env_img_tensor)
            condition_actions.append(torch.from_numpy(env_condition_action))

        # Stack all environments: [num_envs, 3, condition_frame_length, H, W]
        # [8, 3, 5, 256, 256]
        stacked_imgs = torch.stack(img_tensors, dim=0).to(self.device)

        # Reshape to [num_envs, 3, 1, condition_frame_length, H, W] for compatibility
        # [8, 3, 1, 5, 256, 256]
        self.current_obs = stacked_imgs.unsqueeze(2).to(self.device)
        self.condition_action = torch.stack(condition_actions, dim=0).to(self.device)

        num_envs, c, v, t, h, w = self.current_obs.shape
        assert t == self.condition_frame_length, (
            f"Unexpected current_obs shape: {self.current_obs.shape}, expected {num_envs, c, v, self.condition_frame_length, h, w}"
        )

        # Fill image queues for each environment with per-frame tensors [C, 1, H, W]
        for env_idx in range(num_envs):
            frames = [
                self.current_obs[env_idx, :, 0, t_idx : t_idx + 1, :, :]
                for t_idx in range(self.condition_frame_length)
            ]
            self.image_queue[env_idx] = frames
        self.condition_latents = None

        self._reset_metrics()

        # Initialize action buffer (if needed)
        # Initialize with zeros or from init_ee_pose
        init_actions = []
        for init_ee_pose in init_ee_poses:
            if init_ee_pose is not None:
                init_action = init_ee_pose.flatten()
                init_action = self._fit_action_dim(init_action)
            else:
                init_action = np.zeros(self.action_dim, dtype=np.float32)
            init_actions.append(init_action)

        # Store init_ee_poses
        self.task_descriptions = task_descriptions
        self.init_ee_poses = init_ee_poses
        self.state_proxy = torch.as_tensor(
            np.stack(init_actions), device=self.device, dtype=torch.float32
        )

        # Wrap observation to match libero_env format
        extracted_obs = self._wrap_obs()
        infos = {}

        return extracted_obs, infos

    @torch.no_grad()
    def step(self, actions=None, auto_reset=True):
        raise NotImplementedError("step in Wan Env is not impl, use chunk_step instead")

    def _infer_next_chunk_rewards(self):
        """Predict next reward using the reward model"""
        if self.reward_model is None:
            raise ValueError("Reward model is not loaded")

        # Extract chunk observations
        num_envs, c, v, t, h, w = self.current_obs.shape
        extract_chunk_obs = self.current_obs.permute(
            0, 3, 1, 2, 4, 5
        )  # [num_envs, chunk + condition_frame_length, 3, v, h, w]

        if self.cfg.reward_model.type == "ResnetRewModel":
            extract_chunk_obs = extract_chunk_obs[
                :, -self.wan_predict_frames :, :, :, :, :
            ]  # [num_envs, wan_predict_frames, 3, v, h, w]
            extract_chunk_obs = extract_chunk_obs.reshape(
                self.num_envs * self.wan_predict_frames, 3, v, h, w
            )
            extract_chunk_obs = extract_chunk_obs.squeeze(
                2
            )  # [num_envs * wan_predict_frames, 3, h, w]
            extract_chunk_obs = self._select_reward_model_view(extract_chunk_obs)
            extract_chunk_obs = extract_chunk_obs.to(self.device)

            rewards = self.reward_model.predict_rew(extract_chunk_obs)
            rewards = rewards.reshape(self.num_envs, self.wan_predict_frames)
            rewards = self._expand_wan_rewards_to_chunk(rewards)
        elif self.cfg.reward_model.type == "ResNetRewardModel":
            extract_chunk_obs = extract_chunk_obs[
                :, -self.wan_predict_frames :, :, :, :, :
            ]  # [num_envs, wan_predict_frames, 3, v, h, w]
            extract_chunk_obs = extract_chunk_obs.reshape(
                self.num_envs * self.wan_predict_frames, 3, v, h, w
            )
            extract_chunk_obs = extract_chunk_obs.squeeze(
                2
            )  # [num_envs * wan_predict_frames, 3, h, w]
            extract_chunk_obs = self._select_reward_model_view(extract_chunk_obs)
            extract_chunk_obs = extract_chunk_obs.to(self.device)
            extract_chunk_obs = (extract_chunk_obs.clamp(-1.0, 1.0) + 1.0) / 2.0

            rewards = self.reward_model.compute_reward(
                {"main_images": extract_chunk_obs}
            )
            rewards = rewards.reshape(self.num_envs, self.wan_predict_frames)
            rewards = self._expand_wan_rewards_to_chunk(rewards)
        elif self.cfg.reward_model.type == "TaskEmbedResnetRewModel":
            extract_chunk_obs = extract_chunk_obs[
                :, -self.wan_predict_frames :, :, :, :, :
            ]  # [num_envs, wan_predict_frames, 3, v, h, w]
            extract_chunk_obs = extract_chunk_obs.reshape(
                self.num_envs * self.wan_predict_frames, 3, v, h, w
            )
            extract_chunk_obs = extract_chunk_obs.squeeze(
                2
            )  # [num_envs * wan_predict_frames, 3, h, w]
            extract_chunk_obs = self._select_reward_model_view(extract_chunk_obs)
            extract_chunk_obs = extract_chunk_obs.to(self.device)

            # Prepare instructions for each frame in the chunk
            # Each environment has one task description, repeat it for each frame in the chunk
            instructions = []
            for env_idx in range(self.num_envs):
                task_desc = self.task_descriptions[env_idx]
                instructions.extend([task_desc] * self.wan_predict_frames)

            # Predict rewards with instruction conditioning
            rewards = self.reward_model.predict_rew(extract_chunk_obs, instructions)
            rewards = rewards.reshape(self.num_envs, self.wan_predict_frames)
            rewards = self._expand_wan_rewards_to_chunk(rewards)
        else:
            raise ValueError(f"Unknown reward model type: {self.cfg.reward_model.type}")

        return rewards

    def _select_reward_model_view(self, obs):
        """Prepare full-frame reward model input while removing only bottom padding."""
        if self.image_layout == "vertical_3view_padded_bottom":
            valid_height = self.view_height * self.num_views
            if obs.shape[-2] < valid_height:
                raise ValueError(
                    f"Cannot remove bottom padding from reward input height {obs.shape[-2]}; "
                    f"need at least {valid_height}"
                )
            obs = obs[:, :, :valid_height, :]
        return self._resize_reward_model_input(obs)

    def _resize_reward_model_input(self, obs):
        """Optionally resize reward model input while preserving BCHW layout."""
        if obs.ndim != 4:
            raise ValueError(
                "Reward model input must be BCHW before resize, "
                f"got shape {tuple(obs.shape)}"
            )
        if obs.shape[1] != 3:
            raise ValueError(
                "Reward model input must be channel-first with 3 channels, "
                f"got shape {tuple(obs.shape)}"
            )

        resize_cfg = self.cfg.reward_model.get("resize", {})
        if not resize_cfg or not resize_cfg.get("enabled", False):
            return obs

        size = resize_cfg.get("size", None)
        if size is None or len(size) != 2:
            raise ValueError(
                "reward_model.resize.size must be [height, width] when resize is enabled"
            )
        target_size = (int(size[0]), int(size[1]))
        if tuple(obs.shape[-2:]) == target_size:
            return obs

        mode = resize_cfg.get("mode", "bilinear")
        interpolate_kwargs = {"size": target_size, "mode": mode}
        if mode in {"linear", "bilinear", "bicubic", "trilinear"}:
            interpolate_kwargs["align_corners"] = resize_cfg.get(
                "align_corners", False
            )
        return F.interpolate(obs, **interpolate_kwargs)

    def _image_queue_frame_to_pil(self, frame):
        frame = frame[:, 0].detach().cpu().numpy()  # [3, H, W]
        img = np.transpose(frame, (1, 2, 0))
        if img.max() <= 1.2:
            img = ((img + 1.0) / 2.0 * 255.0).clip(0, 255)
        return Image.fromarray(img.astype(np.uint8))

    def _build_wan_condition_images(self):
        batch_input_image = []
        batch_condition_images = []

        for env_idx in range(self.num_envs):
            # image_queue[0] is the fixed reset reference frame. The remaining
            # frames are autoregressive condition frames from the previous chunk.
            imgs = [
                self._image_queue_frame_to_pil(frame)
                for frame in self.image_queue[env_idx]
            ]
            batch_input_image.append(imgs[0])
            batch_condition_images.append(imgs[1 : self.condition_frame_length])

        return batch_input_image, batch_condition_images

    def _encode_condition_latents(self, batch_input_image, batch_condition_images):
        self.pipe.load_models_to_device(["vae"])
        condition_images = [
            [img] + img4 for img, img4 in zip(batch_input_image, batch_condition_images)
        ]
        condition_video = self.pipe.preprocess_video(
            [
                [frame.resize((self.image_size[1], self.image_size[0])) for frame in frames]
                for frames in condition_images
            ]
        )
        return self.pipe.vae.encode(
            condition_video,
            device=self.pipe.device,
            tiled=False,
        ).to(dtype=self.pipe.torch_dtype, device=self.pipe.device)

    def _infer_next_chunk_frames(self, actions):
        """Predict next frame chunk using the wan model"""
        num_envs = self.num_envs
        assert actions.shape[0] == self.num_envs, (
            f"Actions shape {actions.shape} does not match num_envs {self.num_envs}"
        )

        # Normalize actions
        actions_tensor = (
            torch.from_numpy(actions).to(self.device)
            if isinstance(actions, np.ndarray)
            else actions.to(self.device)
        )
        if actions_tensor.shape[-1] != self.action_dim:
            raise ValueError(
                f"Expected action dim {self.action_dim}, got {actions_tensor.shape[-1]}"
            )
        if actions_tensor.shape[1] < self.chunk:
            raise ValueError(
                f"Expected at least {self.chunk} actions for Wan execution, got {actions_tensor.shape[1]}"
            )
        actions_tensor = actions_tensor[:, : self.chunk, :]
        self.condition_action = self.condition_action.to(
            device=actions_tensor.device, dtype=actions_tensor.dtype
        )
        self.condition_action[:, 0, :] = 0
        self.condition_action[:, 0, -1] = -1

        if self.retain_action:
            downsampled_actions = self._downsample_actions_for_wan(actions_tensor)
            actions_tensor = torch.cat(
                [self.condition_action, downsampled_actions], dim=1
            )
        else:
            actions_tensor = self._downsample_actions_for_wan(actions_tensor)

        self.condition_action[:, 1 : self.condition_frame_length, :] = actions_tensor[
            :, -(self.condition_frame_length - 1) :, :
        ]
        # print(f'actions_tensor:{actions_tensor.shape}')
        # Process each environment separately
        all_samples = []

        B = num_envs

        batch_input_image, batch_condition_images = self._build_wan_condition_images()

        condition_latents = None
        if self.use_latent_condition_cache:
            if self.condition_latents is None:
                self.condition_latents = self._encode_condition_latents(
                    batch_input_image, batch_condition_images
                )
            condition_latents = self.condition_latents

        kwargs = {
            "seed": 0,
            "tiled": False,
            "input_image": batch_input_image,  # List[PIL], len = B
            "input_image4": batch_condition_images,
            "condition_latents": condition_latents,
            "action": actions_tensor,
            "height": self.image_size[0],
            "width": self.image_size[1],
            "num_frames": self.num_frames,
            "num_inference_steps": self.num_inference_steps,
            "cfg_scale": 1.0,
            "progress_bar_cmd": lambda x: x,
            "batch_size": B,
            "bs_1": False,
            "return_latents": self.use_latent_condition_cache,
        }

        pipe_output = self.pipe(**kwargs)
        if self.use_latent_condition_cache:
            output, last_latents = pipe_output
        else:
            output = pipe_output
            last_latents = None

        for env_idx in range(num_envs):
            frames = []
            for img in output[env_idx]:
                arr = np.array(img) / 255.0
                arr = arr * 2.0 - 1.0
                frames.append(arr)

            video = np.stack(frames, axis=0)  # [T, H, W, 3]
            video = video.transpose(0, 3, 1, 2)  # [T, 3, H, W]
            video = torch.from_numpy(video)
            video = video.transpose(0, 1)  # [3, T, H, W]

            # Keep image_queue[0] as the reset reference frame and update only
            # the autoregressive condition frames with the latest generated frames.
            reference_frame = self.image_queue[env_idx][0]
            num_autoreg_frames = self.condition_frame_length - 1
            start_t = max(0, video.shape[1] - num_autoreg_frames)
            latest_frames = [
                video[:, t_idx : t_idx + 1] for t_idx in range(start_t, video.shape[1])
            ]
            if len(latest_frames) < num_autoreg_frames:
                latest_frames = [latest_frames[0]] * (
                    num_autoreg_frames - len(latest_frames)
                ) + latest_frames
            self.image_queue[env_idx] = [
                reference_frame,
                *latest_frames[-num_autoreg_frames:],
            ]

            all_samples.append(
                video[
                    :,
                    self.condition_frame_length : self.condition_frame_length
                    + self.wan_predict_frames,
                ]
            )

        if self.use_latent_condition_cache:
            first_latent = self.condition_latents[:, :, 0:1]
            last_generated_latent = last_latents[:, :, -1:]
            self.condition_latents = torch.cat(
                [first_latent, last_generated_latent], dim=2
            ).detach()

        # Stack all environments: [num_envs, C, T, H, W]
        x_samples = torch.stack(all_samples, dim=0).to(self.device)

        # Reshape to match current_obs format: [num_envs, C, 1, T, H, W]
        x_samples = x_samples.unsqueeze(2)

        # Update current observation: append new generated frames to the time dimension
        self.current_obs = torch.cat([self.current_obs, x_samples], dim=3)

        max_frames = self.condition_frame_length + self.wan_predict_frames
        if self.current_obs.shape[3] > max_frames:
            self.current_obs = self.current_obs[:, :, :, -max_frames:, :, :]

    def _wrap_obs(self):
        """Wrap observation to match libero_env format"""
        num_envs = self.num_envs

        # Extract the last frame (most recent observation) for each environment
        # self.current_obs is [b, c, v, t, h, w]  v=1 for single view
        b, c, v, t, h, w = self.current_obs.shape
        assert b == num_envs, (
            f"Unexpected current_obs shape: {self.current_obs.shape}, expected {num_envs}"
        )

        last_frame = self.current_obs[
            :, :, 0, -1, :, :
        ]  # [b,3, v, t,h,w] -> [b, 3, 1, h, w] -> [b, 3, h, w]
        # [8, 3, 256, 256]

        full_image = last_frame.permute(0, 2, 3, 1)  # [b, H, W, 3]
        # Denormalize from [-1, 1] to [0, 255]
        full_image = (full_image + 1.0) / 2.0 * 255.0
        full_image = torch.clamp(full_image, 0, 255)
        # print(f'full_image:{full_image.shape}')
        # print(f'image_size:{self.image_size}')
        # Resize to the configured world-model image size if needed.
        if full_image.shape[1:3] != self.image_size:
            # Reshape for interpolation: [num_envs, H, W, 3] -> [num_envs, 3, H, W]
            full_image = full_image.permute(0, 3, 1, 2)  # [num_envs, 3, H, W]
            # Resize using F.interpolate
            full_image = F.interpolate(
                full_image, size=self.image_size, mode="bilinear", align_corners=False
            )
            full_image = full_image.permute(0, 2, 3, 1)

        # Convert to uint8 tensor (keep as tensor, not numpy)
        full_image = full_image.to(torch.uint8)

        if self.image_layout in {"vertical_3view", "vertical_3view_padded_bottom"}:
            if self.image_layout == "vertical_3view_padded_bottom":
                valid_height = self.view_height * self.num_views
                if full_image.shape[1] < valid_height:
                    raise ValueError(
                        f"Cannot split padded 3-view image with height {full_image.shape[1]}; "
                        f"need at least {valid_height}"
                    )
                full_image = full_image[:, :valid_height, :, :]
                view_h = self.view_height
            else:
                view_h = full_image.shape[1] // 3

            main_image = full_image[:, 0:view_h, :, :]
            left_wrist = full_image[:, view_h : 2 * view_h, :, :]
            right_wrist = full_image[:, 2 * view_h : 3 * view_h, :, :]
            wrist_images = torch.stack([left_wrist, right_wrist], dim=1)
        else:
            main_image = full_image
            wrist_images = None

        states = self.state_proxy.to(device=self.device, dtype=torch.float32)
        if states.shape != (num_envs, self.action_dim):
            raise ValueError(
                f"Unexpected state_proxy shape {states.shape}, expected {(num_envs, self.action_dim)}"
            )

        # Get task descriptions
        if hasattr(self, "task_descriptions"):
            task_descriptions = self.task_descriptions
        else:
            task_descriptions = [""] * num_envs

        # Wrap observation - format aligned with libero_env
        obs = {
            "main_images": main_image,
            "wrist_images": wrist_images,
            "states": states,
            "task_descriptions": task_descriptions,  # list of strings
        }

        return obs

    def capture_image(self):
        """Return full 3-view frames for video recording."""
        b, _c, _v, _t, _h, _w = self.current_obs.shape
        if b != self.num_envs:
            raise ValueError(
                f"Unexpected current_obs shape: {self.current_obs.shape}, "
                f"expected first dim {self.num_envs}"
            )

        last_frame = self.current_obs[:, :, 0, -1, :, :]
        full_image = last_frame.permute(0, 2, 3, 1)
        full_image = (full_image + 1.0) / 2.0 * 255.0
        full_image = torch.clamp(full_image, 0, 255)

        if full_image.shape[1:3] != self.image_size:
            full_image = full_image.permute(0, 3, 1, 2)
            full_image = F.interpolate(
                full_image, size=self.image_size, mode="bilinear", align_corners=False
            )
            full_image = full_image.permute(0, 2, 3, 1)

        full_image = full_image.to(torch.uint8)
        if self.image_layout == "vertical_3view_padded_bottom":
            valid_height = self.view_height * self.num_views
            if full_image.shape[1] < valid_height:
                raise ValueError(
                    f"Cannot capture padded 3-view image with height {full_image.shape[1]}; "
                    f"need at least {valid_height}"
                )
            full_image = full_image[:, :valid_height, :, :]

        return full_image

    def _fit_action_dim(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] < self.action_dim:
            action = np.pad(action, (0, self.action_dim - action.shape[0]))
        elif action.shape[0] > self.action_dim:
            action = action[: self.action_dim]
        return action

    def _handle_auto_reset(self, dones, extracted_obs, infos):
        """Handle automatic reset on episode termination"""
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
        """Execute a chunk of actions - optimized version that processes chunk actions together"""
        # chunk_actions: [num_envs, chunk_steps, action_dim]
        self.onload()
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            self._infer_next_chunk_frames(policy_output_action)

        chunk_actions = (
            torch.from_numpy(policy_output_action).to(self.device)
            if isinstance(policy_output_action, np.ndarray)
            else policy_output_action.to(self.device)
        )
        min_state_action_len = self.action_downsample_stride * self.wan_predict_frames
        if chunk_actions.shape[1] < min_state_action_len:
            raise ValueError(
                f"Need at least {min_state_action_len} actions per chunk to build "
                f"state proxy, got {chunk_actions.shape[1]}"
            )
        self.state_proxy = chunk_actions[
            :, self.action_downsample_stride * self.wan_predict_frames - 1, : self.action_dim
        ].detach().to(
            device=self.device, dtype=torch.float32
        )

        # Update elapsed steps (incremented after inference)
        # print(f'elapsed_steps:{self.elapsed_steps}')
        self.elapsed_steps += self.chunk

        # Read the last frame from self.current_obs
        extracted_obs = self._wrap_obs()

        # Get rewards
        chunk_rewards = self._infer_next_chunk_rewards()
        chunk_rewards_tensors = self._calc_step_reward(chunk_rewards)

        # Estimate success (terminations) based on rewards
        estimated_success = self._estimate_success_from_rewards(chunk_rewards)

        # Create terminations tensor: success is marked at the last step of chunk
        raw_chunk_terminations = torch.zeros(
            self.num_envs, self.chunk, dtype=torch.bool, device=self.device
        )
        raw_chunk_terminations[:, -1] = estimated_success

        raw_chunk_truncations = torch.zeros(
            self.num_envs, self.chunk, dtype=torch.bool, device=self.device
        )
        truncations = torch.tensor(self.elapsed_steps >= self.cfg.max_episode_steps).to(
            self.device
        )

        if truncations.any():
            raw_chunk_truncations[:, -1] = truncations

        past_terminations = raw_chunk_terminations.any(dim=1)
        past_truncations = raw_chunk_truncations.any(dim=1)
        past_dones = torch.logical_or(past_terminations, past_truncations)

        if past_dones.any() and self.auto_reset:
            extracted_obs, infos = self._handle_auto_reset(
                past_dones, extracted_obs, {}
            )
        else:
            infos = {}

        infos = self._record_metrics(
            chunk_rewards_tensors.sum(dim=1), past_terminations, infos
        )

        chunk_terminations = torch.zeros_like(raw_chunk_terminations)
        chunk_terminations[:, -1] = past_terminations

        chunk_truncations = torch.zeros_like(raw_chunk_truncations)
        chunk_truncations[:, -1] = past_truncations

        # Get actions and rewards for rendering
        chunk_actions_for_render = policy_output_action
        if isinstance(chunk_actions_for_render, torch.Tensor):
            chunk_actions_for_render = chunk_actions_for_render.detach().cpu().numpy()
        chunk_rewards_for_render = chunk_rewards_tensors.detach().cpu().numpy()

        # Reshape for rendering: [num_envs, chunk, action_dim] -> [chunk, num_envs, action_dim]
        chunk_actions_for_render = chunk_actions_for_render.transpose(1, 0, 2)
        chunk_rewards_for_render = chunk_rewards_for_render.T  # [chunk, num_envs]

        return (
            [extracted_obs],
            chunk_rewards_tensors,
            chunk_terminations,
            chunk_truncations,
            [infos],
        )

    def offload(self):
        """Move heavy models and runtime tensors to CPU."""
        if self._is_offloaded:
            return
        self.pipe.vae = self.pipe.vae.to("cpu")
        self.pipe.dit = self.pipe.dit.to("cpu")
        self.reward_model = self.reward_model.to("cpu")
        self.current_obs = recursive_to_device(self.current_obs, "cpu")
        if self.condition_latents is not None:
            self.condition_latents = self.condition_latents.cpu()
        self.state_proxy = self.state_proxy.cpu()
        self.prev_step_reward = self.prev_step_reward.cpu()
        self.reset_state_ids = self.reset_state_ids.cpu()
        if self.record_metrics:
            self.success_once = self.success_once.cpu()
            self.returns = self.returns.cpu()
        torch.cuda.empty_cache()
        self._is_offloaded = True

    def onload(self):
        """Move models and runtime tensors back to execution device."""
        if not self._is_offloaded:
            return
        self.pipe.dit = self.pipe.dit.to(self.device)
        self.pipe.vae = self.pipe.vae.to(self.device)
        self.reward_model = self.reward_model.to(self.device)
        self.current_obs = recursive_to_device(self.current_obs, self.device)
        if self.condition_latents is not None:
            self.condition_latents = self.condition_latents.to(self.device)
        self.state_proxy = self.state_proxy.to(self.device)
        self.prev_step_reward = self.prev_step_reward.to(self.device)
        self.reset_state_ids = self.reset_state_ids.to(self.device)
        if self.record_metrics:
            self.success_once = self.success_once.to(self.device)
            self.returns = self.returns.to(self.device)
        self._is_offloaded = False

    def get_state(self) -> bytes:
        """Serialize runtime state to CPU bytes buffer for offload."""
        env_state = {
            "current_obs": recursive_to_device(self.current_obs, "cpu")
            if self.current_obs is not None
            else None,
            "condition_latents": self.condition_latents.cpu()
            if self.condition_latents is not None
            else None,
            "task_descriptions": self.task_descriptions,
            "init_ee_poses": self.init_ee_poses,
            "state_proxy": self.state_proxy.cpu(),
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


# PYTHONPATH="/mnt/project_rlinf/jzn/workspace/release/DiffSynth-Studio:$PYTHONPATH" python -m rlinf.envs.world_model.world_model_wan_env
if __name__ == "__main__":
    from pathlib import Path

    from hydra import compose
    from hydra.core.global_hydra import GlobalHydra
    from hydra.initialize import initialize_config_dir

    # # Set required environment variable
    os.environ.setdefault("EMBODIED_PATH", "examples/embodiment")

    repo_root = Path(__file__).resolve().parents[3]

    # Clear any existing Hydra instance
    GlobalHydra.instance().clear()

    config_dir = Path(
        os.environ.get("EMBODIED_CONFIG_DIR", repo_root / "examples/embodiment/config")
    ).resolve()
    config_name = "wan_libero_spatial_grpo_openvlaoft_quick"

    print(f"Loading config: {config_name} from {config_dir}")
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
        cfg_ = compose(config_name=config_name)
        cfg = cfg_["env"]["train"]

    env = WanEnv(cfg, cfg.total_num_envs, seed_offset=0, total_num_processes=1)

    # Reset environment
    for i in range(20):
        obs, info = env.reset()

    print("\nAfter reset:")
    print(f"  obs keys: {list(obs.keys())}")

    # Test 1: chunk_steps = self.chunk
    print("\n" + "-" * 80)

    # chunk_steps = cfg.chunk
    chunk_steps = cfg.chunk
    num_envs = cfg.total_num_envs

    chunk_traj = 1
    zeros_actions = np.zeros((num_envs, chunk_steps, cfg.get("action_dim", 7)))

    for i in range(chunk_traj):
        print(f"Chunk {i} of {chunk_traj}")
        print("-" * 100)
        o, r, te, tr, infos = env.chunk_step(
            zeros_actions[:, i * chunk_steps : (i + 1) * chunk_steps, :]
        )
