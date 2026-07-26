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
    """Return the current online AMP frame."""
    robot: Articulation = env.scene[reference_body_cfg.name]
    reference_body_id = reference_body_cfg.body_ids[0]
    return build_amp_frame(
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
