"""Deep Robotics DR02-Pro asset configuration.

The model parameters match the official ``DeepRoboticsLab/deep_robotics_model``
DR02-Pro asset at commit ``18192847``. The two neck joints remain fixed so the
robot retains the 29 actuated DOFs used by the expert motion dataset.

The upper-body explicit motor parameters are initialized from the official USD
acceleration drive through ``torque_gain = drive_gain * JointEquivalentInertia``
and tuned to symmetric integer gains. The authored equivalent inertia is applied
as joint armature; omitting it makes the torque-controlled wrists numerically
unstable.
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
        # Keep action=0 near the mechanical zero pose, with the expert's
        # symmetric shoulder yaw and a natural elbow bend.
        joint_pos={
            "^(?!.*(?:_elbow_joint|_shoulder_z_joint)$).*$": 0.0,
            "left_shoulder_z_joint": 0.765,
            "right_shoulder_z_joint": -0.765,
            ".*_elbow_joint": 1.25,
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
            stiffness={
                ".*_shoulder_y_joint": 417.0,
                ".*_shoulder_x_joint": 331.0,
                ".*_shoulder_z_joint": 329.0,
                ".*_elbow_joint": 90.0,
            },
            damping={
                ".*_shoulder_[xyz]_joint": 10.0,
                ".*_elbow_joint": 4.0,
            },
            armature={
                ".*_shoulder_y_joint": 0.6664750278,
                ".*_shoulder_x_joint": 0.5299741328,
                ".*_shoulder_z_joint": 0.5262607038,
                ".*_elbow_joint": 0.1438293383,
            },
            friction=0.0,
        ),
        "wrists": DCMotorCfg(
            joint_names_expr=[".*_wrist_[xyz]_joint"],
            effort_limit=50.0,
            saturation_effort=50.0,
            velocity_limit=23.76,
            stiffness={
                ".*_wrist_z_joint": 87.0,
                ".*_wrist_y_joint": 15.0,
                ".*_wrist_x_joint": 11.0,
            },
            damping={
                ".*_wrist_z_joint": 2.0,
                ".*_wrist_[xy]_joint": 1.0,
            },
            armature={
                ".*_wrist_z_joint": 0.1398061365,
                ".*_wrist_y_joint": 0.0234856559,
                ".*_wrist_x_joint": 0.0175240059,
            },
            friction=0.0,
        ),
        "hip_pitch_roll": DCMotorCfg(
            joint_names_expr=[".*_hip_[xy]_joint"],
            effort_limit=363.0,
            saturation_effort=363.0,
            velocity_limit=20.0,
            stiffness=300.0,
            damping=10.0,
            friction=0.0,
        ),
        "knees": DCMotorCfg(
            joint_names_expr=[".*_knee_joint"],
            effort_limit=363.0,
            saturation_effort=363.0,
            velocity_limit=20.0,
            stiffness=300.0,
            damping=15.0,
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
