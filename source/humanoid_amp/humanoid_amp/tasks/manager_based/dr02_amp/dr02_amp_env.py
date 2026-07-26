"""Manager-based DR02 environment with expert-motion services."""

from __future__ import annotations

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
