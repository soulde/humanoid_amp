"""Shared feature construction for online and expert AMP observations."""

from __future__ import annotations

import torch


def quaternion_to_tangent_and_normal(quaternion: torch.Tensor) -> torch.Tensor:
    """Represent a quaternion by its world-frame tangent and normal vectors."""
    from isaaclab.utils.math import quat_apply

    reference_tangent = torch.zeros_like(quaternion[..., :3])
    reference_normal = torch.zeros_like(quaternion[..., :3])
    reference_tangent[..., 0] = 1.0
    reference_normal[..., 2] = 1.0
    tangent = quat_apply(quaternion, reference_tangent)
    normal = quat_apply(quaternion, reference_normal)
    return torch.cat((tangent, normal), dim=-1)


def build_amp_frame(
    joint_positions: torch.Tensor,
    joint_velocities: torch.Tensor,
    root_positions: torch.Tensor,
    root_rotations: torch.Tensor,
    root_linear_velocities: torch.Tensor,
    root_angular_velocities: torch.Tensor,
    key_body_positions: torch.Tensor,
    *,
    joint_position_scale: float = 1.0,
    joint_velocity_scale: float = 0.05,
    root_linear_velocity_scale: float = 2.0,
    root_angular_velocity_scale: float = 0.25,
) -> torch.Tensor:
    """Build one 101-dimensional DR02 AMP frame.

    Physical scaling is applied here so online and expert observations cannot
    accidentally diverge.
    """
    return torch.cat(
        (
            joint_positions * joint_position_scale,
            joint_velocities * joint_velocity_scale,
            root_positions[:, 2:3],
            quaternion_to_tangent_and_normal(root_rotations),
            root_linear_velocities * root_linear_velocity_scale,
            root_angular_velocities * root_angular_velocity_scale,
            (key_body_positions - root_positions.unsqueeze(1)).flatten(start_dim=1),
        ),
        dim=-1,
    )
