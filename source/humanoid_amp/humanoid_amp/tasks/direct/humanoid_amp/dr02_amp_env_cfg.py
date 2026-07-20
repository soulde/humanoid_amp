"""DR02-Pro AMP task configurations."""

import os

from isaaclab.utils import configclass

from humanoid_amp.assets import DR02_PRO_CFG

from .g1_amp_env_cfg import G1AmpEnvCfg, MOTIONS_DIR


@configclass
class DR02ProAmpEnvCfg(G1AmpEnvCfg):
    robot = DR02_PRO_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    reference_body = "base_link"
    key_body_names = [
        "left_shoulder_y_link",
        "right_shoulder_y_link",
        "left_elbow_link",
        "right_elbow_link",
        "right_hip_z_link",
        "left_hip_z_link",
        "right_wrist_x_link",
        "left_wrist_x_link",
        "right_ankle_x_link",
        "left_ankle_x_link",
    ]


@configclass
class DR02ProAmpWalkEnvCfg(DR02ProAmpEnvCfg):
    motion_file = os.path.join(MOTIONS_DIR, "dr02_pro_walk.npz")
