"""Manager-based DR02 AMP environment configuration."""

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from humanoid_amp.assets import DR02_PRO_CFG

from . import mdp

MOTIONS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../direct/humanoid_amp/motions")
)
REFERENCE_BODY = "base_link"
KEY_BODY_NAMES = [
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

AMP_FRAME_DIM = 101
AMP_HISTORY_LENGTH = 2

def amp_frame_params() -> dict:
    """Create independent scene selectors for an AMP observation term."""
    return {
        "reference_body_cfg": SceneEntityCfg("robot", body_names=[REFERENCE_BODY]),
        "key_bodies_cfg": SceneEntityCfg("robot", body_names=KEY_BODY_NAMES, preserve_order=True),
        "joint_position_scale": 1.0,
        "joint_velocity_scale": 0.05,
        "root_linear_velocity_scale": 2.0,
        "root_angular_velocity_scale": 0.25,
    }


@configclass
class DR02AmpSceneCfg(InteractiveSceneCfg):
    """Flat-ground DR02 scene."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )
    robot = DR02_PRO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )


@configclass
class ActionsCfg:
    """DR02 action specifications."""

    joint_pos = mdp.AmpJointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=1.0,
        use_default_offset=False,
    )


@configclass
class ObservationsCfg:
    """Actor, critic and discriminator observation groups."""

    @configclass
    class PolicyCfg(ObsGroup):
        frame = ObsTerm(func=mdp.amp_frame, params=amp_frame_params())

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        frame = ObsTerm(func=mdp.amp_frame, params=amp_frame_params())

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class AmpCfg(ObsGroup):
        """Two chronological frames: previous, then current."""

        frame = ObsTerm(
            func=mdp.amp_frame,
            params=amp_frame_params(),
            history_length=AMP_HISTORY_LENGTH,
            flatten_history_dim=True,
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    amp: AmpCfg = AmpCfg()


@configclass
class EventsCfg:
    """Expert-motion reset events."""

    reset_from_motion = EventTerm(
        func=mdp.reset_from_expert_motion,
        mode="reset",
    )


@configclass
class RewardsCfg:
    """Task rewards mixed with AMP style reward by the RSL-RL algorithm."""

    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.1)
    joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=0.0)


@configclass
class TerminationsCfg:
    """Episode termination conditions."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    root_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.5})


@configclass
class DR02ProAmpManagerEnvCfg(ManagerBasedRLEnvCfg):
    """Manager-based DR02 AMP environment."""

    scene: DR02AmpSceneCfg = DR02AmpSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    motion_files: list[str] = [os.path.join(MOTIONS_DIR, "dr02_pro_walk.npz")]
    reference_body: str = REFERENCE_BODY
    key_body_names: list[str] = KEY_BODY_NAMES

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 10.0
        self.sim.dt = 1.0 / 60.0
        self.sim.render_interval = self.decimation
        self.sim.physx.gpu_found_lost_pairs_capacity = 2**23
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 2**23
