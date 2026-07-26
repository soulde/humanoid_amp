# RSL-RL AMP development

The ManagerBased AMP implementation uses an editable checkout of
[`leggedrobotics/rsl_rl`](https://github.com/leggedrobotics/rsl_rl).

The compatibility baseline is RSL-RL `v5.0.1`:

```text
3ac56acd3376f2952eb636a133f4b5aa30142552
```

The current AMP implementation tip is:

```text
7b511b1
```

Clone and install it into the Isaac Lab Python environment:

```bash
git clone https://github.com/leggedrobotics/rsl_rl.git third_party/rsl_rl
git -C third_party/rsl_rl switch -c feat/manager-based-amp v5.0.1
/home/soulde/env_isaaclab/bin/python -m pip install --no-deps -e third_party/rsl_rl
```

The checkout is intentionally ignored by the parent repository because it has
its own Git history. Commit AMP algorithm changes inside `third_party/rsl_rl`;
commit environment and integration changes in this repository.
