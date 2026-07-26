"""Pure tensor tests for the manager-based AMP feature layout."""

import importlib.util
from pathlib import Path
import sys
import types

import torch

MODULE_PATH = (
    Path(__file__).parents[1]
    / "humanoid_amp/tasks/manager_based/dr02_amp/amp_features.py"
)


def _load_module():
    fake_math = types.ModuleType("isaaclab.utils.math")

    def quat_apply(quaternion, vector):
        # Identity quaternions are sufficient for the feature-layout test.
        assert torch.allclose(quaternion[:, 0], torch.ones_like(quaternion[:, 0]))
        return vector

    fake_math.quat_apply = quat_apply
    sys.modules.setdefault("isaaclab", types.ModuleType("isaaclab"))
    sys.modules.setdefault("isaaclab.utils", types.ModuleType("isaaclab.utils"))
    sys.modules["isaaclab.utils.math"] = fake_math
    spec = importlib.util.spec_from_file_location("manager_amp_features", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dr02_amp_frame_layout_and_scaling():
    features = _load_module()
    batch = 2
    joint_pos = torch.ones(batch, 29)
    joint_vel = torch.full((batch, 29), 2.0)
    root_pos = torch.tensor([[0.0, 0.0, 0.9], [1.0, 2.0, 1.0]])
    root_quat = torch.zeros(batch, 4)
    root_quat[:, 0] = 1.0
    root_lin_vel = torch.ones(batch, 3)
    root_ang_vel = torch.ones(batch, 3)
    key_body_pos = root_pos.unsqueeze(1) + torch.ones(batch, 10, 3)

    frame = features.build_amp_frame(
        joint_pos,
        joint_vel,
        root_pos,
        root_quat,
        root_lin_vel,
        root_ang_vel,
        key_body_pos,
    )

    assert frame.shape == (batch, 101)
    assert torch.allclose(frame[:, :29], torch.ones(batch, 29))
    assert torch.allclose(frame[:, 29:58], torch.full((batch, 29), 0.1))
    assert torch.allclose(frame[:, 58], root_pos[:, 2])
    assert torch.allclose(frame[:, 65:68], torch.full((batch, 3), 2.0))
    assert torch.allclose(frame[:, 68:71], torch.full((batch, 3), 0.25))
    assert torch.allclose(frame[:, 71:], torch.ones(batch, 30))
