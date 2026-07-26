"""Observation terms for DR02 AMP."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from ..amp_features import build_amp_frame

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def amp_frame(
    env: ManagerBasedEnv,
    reference_body_cfg: SceneEntityCfg,
    key_bodies_cfg: SceneEntityCfg,
    joint_position_scale: float,
    joint_velocity_scale: float,
    root_linear_velocity_scale: float,
    root_angular_velocity_scale: float,
) -> torch.Tensor:
    """Return the current online AMP frame, shared by all observation groups."""
    simulation_step = env._sim_step_counter
    if getattr(env, "_amp_frame_cache_step", None) == simulation_step:
        return env._amp_frame_cache

    robot: Articulation = env.scene[reference_body_cfg.name]
    reference_body_id = reference_body_cfg.body_ids[0]
    frame = build_amp_frame(
        robot.data.joint_pos,
        robot.data.joint_vel,
        robot.data.body_pos_w[:, reference_body_id],
        robot.data.body_quat_w[:, reference_body_id],
        robot.data.body_lin_vel_w[:, reference_body_id],
        robot.data.body_ang_vel_w[:, reference_body_id],
        robot.data.body_pos_w[:, key_bodies_cfg.body_ids],
        joint_position_scale=joint_position_scale,
        joint_velocity_scale=joint_velocity_scale,
        root_linear_velocity_scale=root_linear_velocity_scale,
        root_angular_velocity_scale=root_angular_velocity_scale,
    )
    env._amp_frame_cache = frame
    env._amp_frame_cache_step = simulation_step
    return frame
