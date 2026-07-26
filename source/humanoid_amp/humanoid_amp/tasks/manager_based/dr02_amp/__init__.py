"""Manager-based DR02 AMP task."""

import gymnasium as gym

from . import agents


gym.register(
    id="Isaac-DR02-Pro-AMP-Walk-ManagerBased-v0",
    entry_point=f"{__name__}.dr02_amp_env:DR02ProAmpManagerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dr02_amp_env_cfg:DR02ProAmpManagerEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_amp_cfg:DR02AmpRunnerCfg",
    },
)
