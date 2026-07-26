"""RSL-RL AMP configuration for manager-based DR02."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class RslRlAmpAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Configuration fields consumed by ``rsl_rl.algorithms.AMP``."""

    class_name: str = "AMP"
    discriminator_hidden_dims: list[int] = [1024, 512]
    discriminator_activation: str = "relu"
    discriminator_learning_rate: float = 5.0e-5
    discriminator_batch_size: int = 4096
    discriminator_updates: int = 4
    discriminator_loss_scale: float = 5.0
    discriminator_logit_regularization_scale: float = 0.05
    discriminator_gradient_penalty_scale: float = 5.0
    discriminator_weight_decay_scale: float = 1.0e-4
    amp_replay_buffer_size: int = 200_000
    task_reward_scale: float = 0.0
    style_reward_scale: float = 1.0


@configclass
class DR02AmpRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Training configuration aligned with the existing DR02 skrl AMP run."""

    seed = 42
    device = "cuda:0"
    num_steps_per_env = 16
    max_iterations = 12_500
    save_interval = 625
    experiment_name = "dr02_pro_manager_amp"
    empirical_normalization = False
    clip_actions = None
    obs_groups = {
        "actor": ["policy"],
        "critic": ["critic"],
        "discriminator": ["amp"],
    }

    actor = RslRlMLPModelCfg(
        hidden_dims=[1024, 512],
        activation="relu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=0.055,
            std_type="log",
        ),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[1024, 512],
        activation="relu",
        obs_normalization=True,
    )
    algorithm = RslRlAmpAlgorithmCfg(
        value_loss_coef=2.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=6,
        num_mini_batches=2,
        learning_rate=5.0e-5,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
