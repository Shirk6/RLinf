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
"""LeRobot Agilex (piper) data config for openpi pi0.5 SFT checkpoints.

Ported from OpenDriveLab/kai0 (``src/openpi/training/config.py``
``LerobotAgilexDataConfig``). All public piper SFT entries in kai0 set
``use_delta_joint_actions=False`` (absolute joints), so this is the default
here too. Flip to True if your specific SFT was trained on deltas.
"""
import dataclasses
import pathlib
from typing import Sequence

import openpi.models.model as _model
import openpi.transforms as _transforms
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory
from typing_extensions import override

from rlinf.models.embodiment.openpi.policies import agilex_policy


@dataclasses.dataclass(frozen=True)
class LerobotAgilexDataConfig(DataConfigFactory):
    """Data configuration for the Agilex / piper dual-arm robot."""

    # Convert joint dimensions to deltas before training. Public piper SFTs
    # in kai0 use False (absolute joints).
    use_delta_joint_actions: bool = False

    # Default prompt injected when "prompt" is not in the input data.
    default_prompt: str | None = None

    # Zero out the state input (kai0's mask_state).
    mask_state: bool = False

    # Optional episode subset.
    episodes: list[int] | None = None

    # Repack: LeRobot column -> internal key.
    repack_transforms: _transforms.Group = dataclasses.field(
        default_factory=lambda: _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {
                            "top_head": "observation.images.top_head",
                            "hand_left": "observation.images.hand_left",
                            "hand_right": "observation.images.hand_right",
                        },
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )

    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        default_prompt = self.default_prompt
        repack_transforms = self.repack_transforms

        # If prompt comes from the task, inject the prompt key into the repack.
        if self.base_config and self.base_config.prompt_from_task:
            default_prompt = None
            original_repack = self.repack_transforms.inputs[0]
            new_structure = dict(original_repack.structure)
            new_structure["prompt"] = "prompt"
            repack_transforms = _transforms.Group(
                inputs=[_transforms.RepackTransform(new_structure)]
            )

        data_transforms = _transforms.Group(
            inputs=[
                agilex_policy.AgilexInputs(
                    action_dim=model_config.action_dim,
                    model_type=model_config.model_type,
                    mask_state=self.mask_state,
                )
            ],
            outputs=[agilex_policy.AgilexOutputs()],
        )

        # Delta-joint conversion (only if SFT was trained with deltas).
        if self.use_delta_joint_actions:
            # 6 arm joints delta + 1 gripper absolute, dual-arm.
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=default_prompt)(
            model_config
        )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
            episodes=self.episodes,
        )
