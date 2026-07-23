"""Replay a DR02 AMP motion directly in Isaac Lab, without a policy."""

import argparse
import time

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--motion", required=True, help="AMP .npz motion file")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg

from humanoid_amp.assets import DR02_PRO_CFG
from humanoid_amp.tasks.direct.humanoid_amp.motions import MotionLoader


def main() -> None:
    motion = MotionLoader(args_cli.motion, device=args_cli.device)
    root_index = motion.get_body_index(["base_link"])[0]

    # This viewer is a kinematic data diagnostic. Disable gravity and command the
    # same joint pose that is written below so physics/PD cannot pull the robot
    # away from the reference between two displayed frames.
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=motion.dt, render_interval=1, device=args_cli.device, gravity=(0.0, 0.0, 0.0))
    )
    sim.set_camera_view((3.0, 3.0, 2.0), (0.0, 0.0, 0.8))

    scene_cfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.0)
    scene_cfg.robot = DR02_PRO_CFG.replace(prim_path="/World/Robot")
    scene = InteractiveScene(scene_cfg)
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2000.0).func("/World/Light", sim_utils.DomeLightCfg(intensity=2000.0))

    sim.reset()
    scene.reset()
    robot = scene["robot"]
    indices = motion.get_dof_index(robot.joint_names)
    env_ids = torch.tensor([0], device=args_cli.device)

    print(f"Robot joints ({len(robot.joint_names)}): {robot.joint_names}")
    print(f"Motion joints ({len(motion.dof_names)}): {motion.dof_names}")
    print(f"Motion-to-robot indices: {indices}")
    print("Starting direct DR02 reference-motion replay (policy and actuators bypassed)")

    while simulation_app.is_running():
        for frame in range(motion.num_frames):
            if not simulation_app.is_running():
                return
            joint_pos = motion.dof_positions[frame, indices].unsqueeze(0)
            joint_vel = motion.dof_velocities[frame, indices].unsqueeze(0)
            root_pose = torch.cat(
                (motion.body_positions[frame, root_index], motion.body_rotations[frame, root_index])
            ).unsqueeze(0)
            root_velocity = torch.cat(
                (motion.body_linear_velocities[frame, root_index], motion.body_angular_velocities[frame, root_index])
            ).unsqueeze(0)
            robot.write_root_link_pose_to_sim(root_pose, env_ids)
            robot.write_root_com_velocity_to_sim(root_velocity, env_ids)
            robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
            scene.write_data_to_sim()
            # Refresh articulation transforms and the viewport without advancing
            # PhysX. This is a pure pose viewer, so contacts, PD gains and gravity
            # cannot introduce motion that is absent from the NPZ data.
            sim.forward()
            sim.render()
            time.sleep(motion.dt)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
