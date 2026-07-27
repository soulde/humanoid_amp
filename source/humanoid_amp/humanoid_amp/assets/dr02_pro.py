"""Deep Robotics DR02-Pro asset configuration.

The joint gains and effort limits mirror the position actuators in the
``soulde_robot_zoo`` DR02-Pro MJCF model (commit 57def56).
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets import ArticulationCfg

_ASSET_DIR = os.path.join(os.path.dirname(__file__), "dr02_pro_description")
_URDF_PATH = os.path.join(_ASSET_DIR, "urdf", "dr02_pro.urdf")


def _require_user_tmp_dir() -> str:
    tmp_dir = os.environ.get("TMPDIR")
    if not tmp_dir:
        raise RuntimeError(
            "TMPDIR must be set to a user-specific directory before importing humanoid_amp.assets "
            "(for example, /home/<user>/tmp)."
        )

    return os.path.realpath(os.path.abspath(os.path.expanduser(tmp_dir)))


_USER_TMP_DIR = _require_user_tmp_dir()
_USD_CACHE_DIR = os.path.join(os.path.abspath(os.path.expanduser(_USER_TMP_DIR)), "IsaacLab", "dr02_pro")


DR02_PRO_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=_URDF_PATH,
        fix_base=False,
        merge_fixed_joints=False,
        replace_cylinders_with_capsules=False,
        activate_contact_sensors=True,
        usd_dir=_USD_CACHE_DIR,
        usd_file_name="dr02_pro.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.95),
        # Left-right symmetrized center pose of the DR02 walking expert. This
        # removes gait-phase bias while making action=0 a natural posture for
        # the default + 0.25 * action control mapping.
        joint_pos={
            "waist_z_joint": 0.0,
            "waist_x_joint": 0.0,
            "waist_y_joint": 0.23800436,
            "left_shoulder_y_joint": -0.04193548,
            "left_shoulder_x_joint": -0.06941273,
            "left_shoulder_z_joint": 0.76451367,
            "left_elbow_joint": 1.23470247,
            "left_wrist_z_joint": -0.18035456,
            "left_wrist_y_joint": 0.08305796,
            "left_wrist_x_joint": 0.12865296,
            "right_shoulder_y_joint": -0.04193548,
            "right_shoulder_x_joint": 0.06941273,
            "right_shoulder_z_joint": -0.76451367,
            "right_elbow_joint": 1.23470247,
            "right_wrist_z_joint": 0.18035456,
            "right_wrist_y_joint": 0.08305796,
            "right_wrist_x_joint": -0.12865296,
            "left_hip_y_joint": 0.05838629,
            "left_hip_x_joint": -0.04958252,
            "left_hip_z_joint": 0.12432964,
            "left_knee_joint": -0.00866809,
            "left_ankle_y_joint": 0.04901473,
            "left_ankle_x_joint": 0.05788494,
            "right_hip_y_joint": 0.05838629,
            "right_hip_x_joint": 0.04958252,
            "right_hip_z_joint": -0.12432964,
            "right_knee_joint": -0.00866809,
            "right_ankle_y_joint": 0.04901473,
            "right_ankle_x_joint": -0.05788494,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "waist_yaw": DCMotorCfg(
            joint_names_expr=["waist_z_joint"],
            effort_limit=137.0,
            saturation_effort=137.0,
            velocity_limit=19.38,
            stiffness=200.0,
            damping=10.0,
            friction=0.0,
        ),
        "waist_roll": DCMotorCfg(
            joint_names_expr=["waist_x_joint"],
            effort_limit=137.0,
            saturation_effort=137.0,
            velocity_limit=19.38,
            stiffness=2800.0,
            damping=15.0,
            friction=0.0,
        ),
        "waist_pitch": DCMotorCfg(
            joint_names_expr=["waist_y_joint"],
            effort_limit=363.0,
            saturation_effort=363.0,
            velocity_limit=20.0,
            stiffness=2300.0,
            damping=20.0,
            friction=0.0,
        ),
        "arms": DCMotorCfg(
            joint_names_expr=[".*_shoulder_[xyz]_joint", ".*_elbow_joint"],
            effort_limit=137.0,
            saturation_effort=137.0,
            velocity_limit=19.38,
            stiffness=100.0,
            damping=5.0,
            friction=0.0,
        ),
        "wrists": DCMotorCfg(
            joint_names_expr=[".*_wrist_[xyz]_joint"],
            effort_limit=50.0,
            saturation_effort=50.0,
            velocity_limit=23.76,
            stiffness=90.0,
            damping=2.0,
            friction=0.0,
        ),
        "leg_pitch_roll_knee": DCMotorCfg(
            joint_names_expr=[".*_hip_[xy]_joint", ".*_knee_joint"],
            effort_limit=363.0,
            saturation_effort=363.0,
            velocity_limit=20.0,
            stiffness=300.0,
            damping=10.0,
            friction=0.0,
        ),
        "hip_yaw": DCMotorCfg(
            joint_names_expr=[".*_hip_z_joint"],
            effort_limit=137.0,
            saturation_effort=137.0,
            velocity_limit=19.38,
            stiffness=300.0,
            damping=10.0,
            friction=0.0,
        ),
        "ankle_pitch": DCMotorCfg(
            joint_names_expr=[".*_ankle_y_joint"],
            effort_limit=137.0,
            saturation_effort=137.0,
            velocity_limit=19.38,
            stiffness=80.0,
            damping=3.0,
            friction=0.0,
        ),
        "ankle_roll": DCMotorCfg(
            joint_names_expr=[".*_ankle_x_joint"],
            effort_limit=50.0,
            saturation_effort=50.0,
            velocity_limit=23.76,
            stiffness=30.0,
            damping=1.0,
            friction=0.0,
        ),
    },
)
