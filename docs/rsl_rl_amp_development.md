# RSL-RL AMP development

The ManagerBased AMP implementation uses an editable checkout of
[`soulde/rsl_rl`](https://github.com/soulde/rsl_rl). The official
[`leggedrobotics/rsl_rl`](https://github.com/leggedrobotics/rsl_rl) repository
is configured as the `upstream` remote.

The compatibility baseline is RSL-RL `v5.0.1`:

```text
3ac56acd3376f2952eb636a133f4b5aa30142552
```

The current AMP implementation tip is:

```text
7b511b1db872bcb60ff88830138ffb2163e94910
```

Clone and install it into the Isaac Lab Python environment:

```bash
git clone --branch feat/manager-based-amp https://github.com/soulde/rsl_rl.git third_party/rsl_rl
git -C third_party/rsl_rl remote add upstream https://github.com/leggedrobotics/rsl_rl.git
/home/soulde/env_isaaclab/bin/python -m pip install --no-deps -e third_party/rsl_rl
```

The checkout is intentionally ignored by the parent repository because it has
its own Git history. Commit AMP algorithm changes inside `third_party/rsl_rl`;
commit environment and integration changes in this repository.
