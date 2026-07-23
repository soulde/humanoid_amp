# Project Agent Instructions

## Long-running training

- Run every long-running training job inside a named `tmux` session.
- Use a descriptive session name that identifies the task and training budget.
- Redirect the training output to a persistent log file under `logs/tmux/`.
- After launch, report the tmux session name, training PID, experiment directory,
  and log path.
- Before starting a replacement training job, resolve and stop the exact prior
  play or training process so jobs do not compete for GPU resources.
