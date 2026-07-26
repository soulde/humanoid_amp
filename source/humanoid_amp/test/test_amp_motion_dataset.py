"""Tests for multi-file AMP expert sampling."""

from pathlib import Path
import sys
import types

import numpy as np
import torch

PACKAGE_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

# Avoid importing the package-level Isaac Lab task registry in this tensor-only test.
package_paths = {
    "humanoid_amp": "humanoid_amp",
    "humanoid_amp.tasks": "humanoid_amp/tasks",
    "humanoid_amp.tasks.direct": "humanoid_amp/tasks/direct",
    "humanoid_amp.tasks.direct.humanoid_amp": "humanoid_amp/tasks/direct/humanoid_amp",
    "humanoid_amp.tasks.manager_based": "humanoid_amp/tasks/manager_based",
    "humanoid_amp.tasks.manager_based.dr02_amp": "humanoid_amp/tasks/manager_based/dr02_amp",
}
for package_name, relative_path in package_paths.items():
    fake_package = types.ModuleType(package_name)
    fake_package.__path__ = [str(PACKAGE_ROOT / relative_path)]
    sys.modules.setdefault(package_name, fake_package)

from humanoid_amp.tasks.manager_based.dr02_amp.motion_dataset import AmpMotionDataset

MOTION_FILE = (
    PACKAGE_ROOT
    / "humanoid_amp/tasks/direct/humanoid_amp/motions/dr02_pro_walk.npz"
)


def test_multiple_npz_files_produce_chronological_amp_pairs(monkeypatch):
    fake_math = types.ModuleType("isaaclab.utils.math")
    fake_math.quat_apply = lambda quaternion, vector: vector
    sys.modules.setdefault("isaaclab", types.ModuleType("isaaclab"))
    sys.modules.setdefault("isaaclab.utils", types.ModuleType("isaaclab.utils"))
    sys.modules["isaaclab.utils.math"] = fake_math

    data = np.load(MOTION_FILE)
    dataset = AmpMotionDataset(
        [str(MOTION_FILE), str(MOTION_FILE)],
        "cpu",
        data["dof_names"].tolist(),
        data["body_names"].tolist(),
        "base_link",
        data["body_names"].tolist()[1:],
    )
    monkeypatch.setattr(dataset, "_sample_motion_ids", lambda count: np.arange(count) % 2)

    observations = dataset.sample_amp_observations(4)

    assert dataset.num_motions == 2
    assert observations.shape == (4, 202)
    assert torch.isfinite(observations).all()
