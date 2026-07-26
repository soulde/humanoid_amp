"""Action terms for DR02 AMP."""

from __future__ import annotations

from isaaclab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from isaaclab.managers import ActionTerm
from isaaclab.utils import configclass


class AmpJointPositionAction(JointPositionAction):
    """Map normalized actions exactly like the original Direct AMP task.

    The original task uses the midpoint of each soft joint range as its offset
    and the complete range as its scale.
    """

    def __init__(self, cfg: "AmpJointPositionActionCfg", env):
        super().__init__(cfg, env)
        limits = self._asset.data.soft_joint_pos_limits[0, self._joint_ids]
        self._offset = 0.5 * (limits[:, 0] + limits[:, 1])
        self._scale = limits[:, 1] - limits[:, 0]


@configclass
class AmpJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for :class:`AmpJointPositionAction`."""

    class_type: type[ActionTerm] = AmpJointPositionAction
