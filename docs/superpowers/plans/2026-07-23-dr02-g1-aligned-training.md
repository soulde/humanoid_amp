# DR02 G1-Aligned Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve the current DR02 experiment on `exp/dr02_task`, align the DR02 agent configuration with G1 walk, and launch a 200,000-timestep headless training run.

**Architecture:** Save the complete dirty working tree as a reproducible experiment commit on a dedicated branch. Return to `dev`, copy the G1 agent configuration into the DR02 configuration while applying only the DR02 log-directory and timestep overrides, verify the YAML structurally, then stop playback and launch training.

**Tech Stack:** Git, YAML, Python 3.11, Isaac Lab, skrl AMP

## Global Constraints

- Preserve all current DR02 changes on `exp/dr02_task`.
- Use `dev` as the main DR02 development branch because `master` predates the DR02 task.
- The final DR02 config differs from G1 walk only at `agent.experiment.directory: dr02_pro_amp_walk` and `trainer.timesteps: 200000`.
- Use `TMPDIR=/home/soulde/tmp`.

---

### Task 1: Preserve the current experiment

**Files:**
- Commit: all currently modified and untracked DR02 files

**Interfaces:**
- Consumes: current dirty `dev` working tree
- Produces: branch `exp/dr02_task` with a reproducible experiment commit

- [ ] **Step 1: Create the experiment branch**

Run: `git switch -c exp/dr02_task`
Expected: branch changes from `dev` to `exp/dr02_task`.

- [ ] **Step 2: Review and commit the current experiment**

Run: `git status --short && git diff --check`
Expected: only the known DR02 environment, config, motion, and replay-tool changes.

Run: `git add source/humanoid_amp/humanoid_amp/tasks/direct/humanoid_amp/__init__.py source/humanoid_amp/humanoid_amp/tasks/direct/humanoid_amp/agents/skrl_dr02_pro_walk_amp_cfg.yaml source/humanoid_amp/humanoid_amp/tasks/direct/humanoid_amp/dr02_amp_env.py source/humanoid_amp/humanoid_amp/tasks/direct/humanoid_amp/dr02_amp_env_cfg.py source/humanoid_amp/humanoid_amp/tasks/direct/humanoid_amp/motions/dr02_pro_walk.npz source/humanoid_amp/humanoid_amp/tasks/direct/humanoid_amp/motions/dr02_pro_walk.pre_torso_foot_fix.20260721.npz scripts/tools/replay_dr02_amp_motion.py`

Run: `git commit -m "experiment: preserve DR02 task configuration"`
Expected: one commit containing the complete current DR02 experiment.

- [ ] **Step 3: Return to the development branch**

Run: `git switch dev`
Expected: clean `dev` working tree.

### Task 2: Align the DR02 agent and launch training

**Files:**
- Modify: `source/humanoid_amp/humanoid_amp/tasks/direct/humanoid_amp/agents/skrl_dr02_pro_walk_amp_cfg.yaml:1`

**Interfaces:**
- Consumes: `skrl_g1_walk_amp_cfg.yaml`
- Produces: a DR02 config matching G1 except for log directory and timestep budget

- [ ] **Step 1: Run the structural assertion before modification**

Run a Python YAML comparison that removes `agent.experiment.directory` and `trainer.timesteps` before comparing both dictionaries.

Expected: FAIL because current DR02 values include `fixed_log_std: False`, `grad_norm_clip: 1.0`, `entropy_loss_scale: 1.0e-03`, and `task_reward_scale: 1.0`.

- [ ] **Step 2: Apply the minimal configuration change**

Set the DR02 file to the exact G1 walk YAML content, then set:

```yaml
agent:
  experiment:
    directory: "dr02_pro_amp_walk"
trainer:
  timesteps: 200000
```

- [ ] **Step 3: Verify the structural assertion passes**

Run the same Python YAML comparison.
Expected: PASS, with the directory equal to `dr02_pro_amp_walk` and timesteps equal to `200000`.

- [ ] **Step 4: Stop WebRTC playback**

Resolve the exact `scripts/skrl/play.py` PID and terminate it. Confirm no playback process remains before training.

- [ ] **Step 5: Launch training**

Run:

```bash
TMPDIR=/home/soulde/tmp /home/soulde/env_isaaclab/bin/python scripts/skrl/train.py \
  --task Isaac-DR02-Pro-AMP-Walk-Direct-v0 \
  --algorithm AMP \
  --headless
```

Expected: Isaac Lab loads the DR02 environment and the skrl trainer starts with `timesteps: 200000`.
