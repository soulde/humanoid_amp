"""Manager-based DR02 environment with expert-motion services."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs import ManagerBasedRLEnv

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
        self._amp_reset_env_ids: torch.Tensor | None = None
        self._amp_reset_previous_frame: torch.Tensor | None = None

    def set_amp_reset_history(self, env_ids: torch.Tensor, previous_frame: torch.Tensor) -> None:
        """Stage preceding expert frames until ObservationManager reset completes."""
        self._amp_reset_env_ids = env_ids.clone()
        self._amp_reset_previous_frame = previous_frame

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)
        if self._amp_reset_env_ids is None or self._amp_reset_previous_frame is None:
            return

        history = self.observation_manager._group_obs_term_history_buffer["amp"]["frame"]
        reset_env_ids = self._amp_reset_env_ids
        previous_frame = self._amp_reset_previous_frame
        if history._buffer is None:
            if len(reset_env_ids) != self.num_envs:
                raise RuntimeError("AMP history must be initialized with a full-environment reset")
            history.append(previous_frame)
        else:
            history._buffer[history._pointer, reset_env_ids] = previous_frame
            history._num_pushes[reset_env_ids] = 1

        self._amp_reset_env_ids = None
        self._amp_reset_previous_frame = None
