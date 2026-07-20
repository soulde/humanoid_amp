"""Convert a GMR robot-motion pickle to the AMP NumPy dataset format."""

import argparse
import pickle
from pathlib import Path

import mujoco
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation


DR02_AMP_BODY_NAMES = [
    "base_link",
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


def finite_difference(values: np.ndarray, dt: float) -> np.ndarray:
    return gaussian_filter1d(np.gradient(values, dt, axis=0), sigma=1, axis=0).astype(np.float32)


def angular_velocity(quaternions: np.ndarray, dt: float) -> np.ndarray:
    velocity = np.zeros((*quaternions.shape[:-1], 3), dtype=np.float32)
    for body_index in range(quaternions.shape[1]):
        rotations = Rotation.from_quat(quaternions[:, body_index], scalar_first=True)
        increments = (rotations[:-1].inv() * rotations[1:]).as_rotvec() / dt
        velocity[:-1, body_index] = increments
        velocity[-1, body_index] = increments[-1]
    return gaussian_filter1d(velocity, sigma=1, axis=0)


def movable_joint_names(model: mujoco.MjModel) -> list[str]:
    return [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        for joint_id in range(model.njnt)
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", type=Path, required=True, help="GMR pickle produced by smplx_to_robot.py")
    parser.add_argument("--mjcf", type=Path, required=True, help="Matching DR02-Pro MJCF model")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.motion.open("rb") as stream:
        motion = pickle.load(stream)

    model = mujoco.MjModel.from_xml_path(str(args.mjcf))
    data = mujoco.MjData(model)
    root_pos = np.asarray(motion["root_pos"], dtype=np.float32)
    root_quat = np.asarray(motion["root_rot"], dtype=np.float32)[:, [3, 0, 1, 2]]
    dof_positions = np.asarray(motion["dof_pos"], dtype=np.float32)
    fps = float(motion["fps"])

    if model.nq != dof_positions.shape[1] + 7:
        raise ValueError(f"Motion has {dof_positions.shape[1]} joints, but MJCF expects {model.nq - 7}")

    body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in DR02_AMP_BODY_NAMES]
    if any(body_id < 0 for body_id in body_ids):
        raise ValueError("MJCF does not contain all DR02 AMP key bodies")

    frame_count = len(root_pos)
    body_positions = np.empty((frame_count, len(body_ids), 3), dtype=np.float32)
    body_rotations = np.empty((frame_count, len(body_ids), 4), dtype=np.float32)
    for frame in range(frame_count):
        data.qpos[:3] = root_pos[frame]
        data.qpos[3:7] = root_quat[frame]
        data.qpos[7:] = dof_positions[frame]
        mujoco.mj_forward(model, data)
        body_positions[frame] = data.xpos[body_ids]
        body_rotations[frame] = data.xquat[body_ids]

    dt = 1.0 / fps
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        fps=np.asarray(fps),
        dof_names=np.asarray(movable_joint_names(model)),
        body_names=np.asarray(DR02_AMP_BODY_NAMES),
        dof_positions=dof_positions,
        dof_velocities=finite_difference(dof_positions, dt),
        body_positions=body_positions,
        body_rotations=body_rotations,
        body_linear_velocities=finite_difference(body_positions, dt),
        body_angular_velocities=angular_velocity(body_rotations, dt),
    )
    print(f"Wrote {frame_count} frames to {args.output}")


if __name__ == "__main__":
    main()
