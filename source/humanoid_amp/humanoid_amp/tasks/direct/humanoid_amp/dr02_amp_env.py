"""DR02-Pro specialization of the G1 AMP environment."""

import torch

from .g1_amp_env import G1AmpEnv


class DR02ProAmpEnv(G1AmpEnv):
    """DR02 entry point using the same AMP behavior as the G1 demo.

    Robot-specific asset, joint/body names, and motion data are supplied by
    :class:`DR02ProAmpEnvCfg`; control and learning behavior stay inherited.
    """

    def _get_rewards(self) -> torch.Tensor:
        rewards = super()._get_rewards()
        reward_log = self.extras["log"]

        # Names containing a slash are forwarded verbatim by skrl's existing
        # environment-info logger and therefore appear as separate TensorBoard
        # scalar groups. No trainer/library changes are needed.
        reward_log["Reward terms / task total"] = rewards.mean()
        reward_log["Reward terms / joint position limits"] = reward_log.pop("rew_joint_pos_limits")
        reward_log["Reward terms / joint acceleration"] = reward_log.pop("rew_joint_acc_l2")
        reward_log["Reward terms / termination"] = reward_log.pop("rew_termination")
        reward_log["Reward terms / action L2"] = reward_log.pop("rew_action_l2")
        reward_log["Reward terms / joint velocity"] = reward_log.pop("rew_joint_vel_l2")

        lower = self.robot.data.soft_joint_pos_limits[:, :, 0]
        upper = self.robot.data.soft_joint_pos_limits[:, :, 1]
        limit_violation = torch.clamp(lower - self.robot.data.joint_pos, min=0.0)
        limit_violation += torch.clamp(self.robot.data.joint_pos - upper, min=0.0)
        reward_log["Reward metrics / joint limit violation (rad)"] = limit_violation.sum(dim=1).mean()
        reward_log["Reward metrics / joint acceleration squared"] = (
            self.robot.data.joint_acc.square().sum(dim=1).mean()
        )
        return rewards

    def _get_observations(self) -> dict:
        # G1 rebuilds ``extras`` to publish AMP observations. Preserve the
        # reward scalars computed earlier in the same environment step.
        reward_log = self.extras.get("log")
        observations = super()._get_observations()
        if reward_log is not None:
            self.extras["log"] = reward_log
        return observations
