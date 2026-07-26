"""Reset events for DR02 AMP."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from ..dr02_amp_env import DR02ProAmpManagerEnv


def reset_from_expert_motion(
    env: DR02ProAmpManagerEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset selected environments to random expert-motion states."""
    robot: Articulation = env.scene[asset_cfg.name]
    state, previous_amp_frame = env.motion_dataset.sample_reset_states(len(env_ids))
    reference_index = env.motion_dataset.body_names.index(env.cfg.reference_body)

    root_pose = torch.cat(
        (
            state.body_positions[:, reference_index] + env.scene.env_origins[env_ids],
            state.body_rotations[:, reference_index],
        ),
        dim=-1,
    )
    root_pose[:, 2] += 0.05
    root_velocity = torch.cat(
        (
            state.body_linear_velocities[:, reference_index],
            state.body_angular_velocities[:, reference_index],
        ),
        dim=-1,
    )
    robot.write_root_link_pose_to_sim(root_pose, env_ids=env_ids)
    robot.write_root_com_velocity_to_sim(root_velocity, env_ids=env_ids)
    robot.write_joint_state_to_sim(
        state.joint_positions,
        state.joint_velocities,
        env_ids=env_ids,
    )
    env.set_amp_reset_history(env_ids, previous_amp_frame)
