# DR02 G1-Aligned Training Design

## Goal

Preserve the current DR02-Pro experiment state on `exp/dr02_task`, then
start a new DR02-Pro training run from `dev` with the G1 walk agent
configuration and a 200,000-timestep budget.

## Branch and state handling

1. Create `exp/dr02_task` from the current `dev` working tree.
2. Commit all current DR02-Pro changes on that branch so the current
   configuration, environment code, replay tool, and motion assets are
   reproducible.
3. Switch back to `dev`.
4. Change only the DR02-Pro agent configuration on `dev`.

The repository's `master` branch is not used because it predates the
DR02-Pro task. In this workflow, `dev` is the main DR02 development branch.

## Training configuration

Make `skrl_dr02_pro_walk_amp_cfg.yaml` identical to
`skrl_g1_walk_amp_cfg.yaml`, except for these intentional differences:

- `agent.experiment.directory` remains `dr02_pro_amp_walk`.
- `trainer.timesteps` is `200000`.

This includes using the G1 values for fixed policy standard deviation,
gradient clipping, entropy loss, task reward, and discriminator
regularization.

## Execution

Stop the current WebRTC policy playback before training so it does not
compete for GPU resources. Start a headless DR02-Pro AMP walk training run
with the repository's required user-specific `TMPDIR`.

## Verification

- Confirm `exp/dr02_task` contains the preserved commit.
- Confirm `dev` contains only the intended DR02 agent-config differences
  relative to G1.
- Parse the resulting YAML and assert the experiment directory and
  200,000-timestep override.
- Confirm the training process reaches environment setup and begins the
  trainer loop.
