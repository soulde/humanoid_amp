"""Observation terms for DR02 AMP."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..dr02_amp_env import DR02ProAmpManagerEnv


def amp_frame(env: DR02ProAmpManagerEnv) -> torch.Tensor:
    """Return the current online AMP frame, shared by all observation groups."""
    return env.compute_amp_frame()
