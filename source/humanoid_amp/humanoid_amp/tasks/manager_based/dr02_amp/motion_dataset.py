"""Multi-file expert motion dataset for DR02 AMP."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence

import numpy as np
import torch

from humanoid_amp.tasks.direct.humanoid_amp.motions import MotionLoader

from .amp_features import build_amp_frame


@dataclass
class MotionState:
    """A batch of expert kinematic states in requested robot ordering."""

    joint_positions: torch.Tensor
    joint_velocities: torch.Tensor
    body_positions: torch.Tensor
    body_rotations: torch.Tensor
    body_linear_velocities: torch.Tensor
    body_angular_velocities: torch.Tensor


class AmpMotionDataset:
    """Sample compatible expert motions from one or more NPZ files."""

    def __init__(
        self,
        motion_files: Sequence[str],
        device: str | torch.device,
        joint_names: Sequence[str],
        body_names: Sequence[str],
        reference_body: str,
        key_body_names: Sequence[str],
    ):
        if not motion_files:
            raise ValueError("At least one AMP motion file is required")

        self.device = torch.device(device)
        self.joint_names = list(joint_names)
        self.body_names = list(body_names)
        self.reference_body = reference_body
        self.key_body_names = list(key_body_names)
        self.loaders = [MotionLoader(str(Path(path).expanduser().resolve()), self.device) for path in motion_files]

        self._joint_indexes: list[list[int]] = []
        self._body_indexes: list[list[int]] = []
        for loader in self.loaders:
            self._joint_indexes.append(loader.get_dof_index(self.joint_names))
            self._body_indexes.append(loader.get_body_index(self.body_names))
            loader.get_body_index([self.reference_body, *self.key_body_names])

        durations = np.asarray([loader.duration for loader in self.loaders], dtype=np.float64)
        if np.any(durations <= 0.0):
            raise ValueError("Every AMP motion must contain at least two frames")
        self._sampling_probabilities = durations / durations.sum()

        self._reference_body_index = self.body_names.index(self.reference_body)
        self._key_body_indexes = [self.body_names.index(name) for name in self.key_body_names]

    @property
    def num_motions(self) -> int:
        return len(self.loaders)

    @property
    def amp_observation_dim(self) -> int:
        return 202

    def _sample_motion_ids(self, num_samples: int) -> np.ndarray:
        return np.random.choice(self.num_motions, size=num_samples, p=self._sampling_probabilities)

    def sample_states(
        self,
        num_samples: int,
        *,
        motion_ids: np.ndarray | None = None,
        times: np.ndarray | None = None,
    ) -> tuple[MotionState, np.ndarray, np.ndarray]:
        """Sample current expert states and return their motion IDs and times."""
        motion_ids = self._sample_motion_ids(num_samples) if motion_ids is None else np.asarray(motion_ids)
        if motion_ids.shape != (num_samples,):
            raise ValueError(f"Expected motion_ids shape {(num_samples,)}, got {motion_ids.shape}")
        if times is not None and np.asarray(times).shape != (num_samples,):
            raise ValueError(f"Expected times shape {(num_samples,)}, got {np.asarray(times).shape}")

        output: list[torch.Tensor | None] = [None] * 6
        sampled_times = np.empty(num_samples, dtype=np.float64)
        for motion_id, loader in enumerate(self.loaders):
            sample_indexes = np.flatnonzero(motion_ids == motion_id)
            if sample_indexes.size == 0:
                continue
            motion_times = (
                loader.sample_times(sample_indexes.size)
                if times is None
                else np.asarray(times, dtype=np.float64)[sample_indexes]
            )
            sampled_times[sample_indexes] = motion_times
            values = list(loader.sample(sample_indexes.size, motion_times))
            values[0] = values[0][:, self._joint_indexes[motion_id]]
            values[1] = values[1][:, self._joint_indexes[motion_id]]
            for index in range(2, 6):
                values[index] = values[index][:, self._body_indexes[motion_id]]

            torch_indexes = torch.as_tensor(sample_indexes, device=self.device)
            for index, value in enumerate(values):
                if output[index] is None:
                    output[index] = torch.empty(
                        (num_samples, *value.shape[1:]), dtype=value.dtype, device=self.device
                    )
                output[index][torch_indexes] = value

        return MotionState(*output), motion_ids, sampled_times

    def sample_amp_observations(self, num_samples: int) -> torch.Tensor:
        """Sample chronological ``previous, current`` discriminator observations."""
        motion_ids = self._sample_motion_ids(num_samples)
        current_times = np.empty(num_samples, dtype=np.float64)
        previous_times = np.empty(num_samples, dtype=np.float64)
        for motion_id, loader in enumerate(self.loaders):
            mask = motion_ids == motion_id
            current_times[mask] = loader.sample_times(int(mask.sum()))
            previous_times[mask] = current_times[mask] - loader.dt

        previous, _, _ = self.sample_states(
            num_samples,
            motion_ids=motion_ids,
            times=previous_times,
        )
        current, _, _ = self.sample_states(
            num_samples,
            motion_ids=motion_ids,
            times=current_times,
        )
        previous_frame = self.build_frames(previous)
        current_frame = self.build_frames(current)
        return torch.cat((previous_frame, current_frame), dim=-1)

    def sample_reset_states(self, num_samples: int) -> tuple[MotionState, torch.Tensor]:
        """Sample reset states and the immediately preceding AMP frame."""
        current, motion_ids, current_times = self.sample_states(num_samples)
        previous_times = current_times.copy()
        for motion_id, loader in enumerate(self.loaders):
            mask = motion_ids == motion_id
            previous_times[mask] -= loader.dt
        previous, _, _ = self.sample_states(
            num_samples,
            motion_ids=motion_ids,
            times=previous_times,
        )
        return current, self.build_frames(previous)

    def build_frames(self, state: MotionState) -> torch.Tensor:
        """Build one AMP frame from a batch of sampled motion states."""
        reference_index = self._reference_body_index
        return build_amp_frame(
            state.joint_positions,
            state.joint_velocities,
            state.body_positions[:, reference_index],
            state.body_rotations[:, reference_index],
            state.body_linear_velocities[:, reference_index],
            state.body_angular_velocities[:, reference_index],
            state.body_positions[:, self._key_body_indexes],
        )
