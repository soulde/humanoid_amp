"""Manager-based DR02 environment with expert-motion services."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs import ManagerBasedRLEnv

from .amp_features import build_amp_frame
from .dr02_amp_env_cfg import DR02ProAmpManagerEnvCfg
from .motion_dataset import AmpMotionDataset


class DR02ProAmpManagerEnv(ManagerBasedRLEnv):
    """Manager-based environment exposing expert samples to RSL-RL AMP."""

    cfg: DR02ProAmpManagerEnvCfg

    def __init__(self, cfg: DR02ProAmpManagerEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        robot = self.scene["robot"]
        self.motion_dataset = AmpMotionDataset(
            cfg.motion_files,
            self.device,
            robot.joint_names,
            [cfg.reference_body, *cfg.key_body_names],
            cfg.reference_body,
            cfg.key_body_names,
        )
        self._resolve_amp_body_ids()

    def _resolve_amp_body_ids(self) -> None:
        """Resolve body IDs lazily because managers are built inside ``super().__init__``."""
        if hasattr(self, "_amp_reference_body_id"):
            return
        robot = self.scene["robot"]
        reference_ids, _ = robot.find_bodies([self.cfg.reference_body], preserve_order=True)
        key_body_ids, _ = robot.find_bodies(self.cfg.key_body_names, preserve_order=True)
        self._amp_reference_body_id = reference_ids[0]
        self._amp_key_body_ids = key_body_ids

    def compute_amp_frame(self) -> torch.Tensor:
        """Compute or reuse the current 101-dimensional AMP state."""
        self._resolve_amp_body_ids()
        if getattr(self, "_amp_frame_cache_step", None) == self._sim_step_counter:
            return self._amp_frame_cache

        robot = self.scene["robot"]
        reference_body_id = self._amp_reference_body_id
        frame = build_amp_frame(
            robot.data.joint_pos,
            robot.data.joint_vel,
            robot.data.body_pos_w[:, reference_body_id],
            robot.data.body_quat_w[:, reference_body_id],
            robot.data.body_lin_vel_w[:, reference_body_id],
            robot.data.body_ang_vel_w[:, reference_body_id],
            robot.data.body_pos_w[:, self._amp_key_body_ids],
        )
        self._amp_frame_cache = frame
        self._amp_frame_cache_step = self._sim_step_counter
        return frame

    def step(self, action: torch.Tensor):
        """Clear stale terminal data before executing the next environment step."""
        self.extras.pop("terminal_amp_observations", None)
        return super().step(action)

    def _reset_idx(self, env_ids: Sequence[int]):
        """Capture terminal AMP states immediately before Isaac Lab resets."""
        if hasattr(self, "_amp_reference_body_id") and len(env_ids) > 0:
            self.extras["terminal_amp_observations"] = self.compute_amp_frame().clone()
        super()._reset_idx(env_ids)
