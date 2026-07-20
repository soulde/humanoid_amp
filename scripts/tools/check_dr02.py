"""Smoke-test the DR02-Pro asset conversion and articulation initialization."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Validate the DR02-Pro Isaac Lab asset.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation

from humanoid_amp.assets import DR02_PRO_CFG


def main() -> None:
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device=args_cli.device))
    sim_utils.GroundPlaneCfg().func("/World/Ground", sim_utils.GroundPlaneCfg())

    robot = Articulation(DR02_PRO_CFG.replace(prim_path="/World/DR02Pro"))
    sim.reset()

    if robot.num_joints != 29:
        raise RuntimeError(f"Expected 29 DR02-Pro joints, found {robot.num_joints}")

    robot.write_data_to_sim()
    sim.step()
    robot.update(sim.get_physics_dt())
    print(f"DR02-Pro validation passed: {robot.num_joints} joints, {robot.num_bodies} bodies")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
