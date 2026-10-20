# Humanoid real2sim commands

This directory contains command line entry points and task specific programs.
Shared runtime code belongs in [`robosimstudio/humanoid_real2sim/`](../../robosimstudio/humanoid_real2sim/README.md). Task descriptions, configuration, assets, motions, and review results belong in [`examples/humanoid-real2sim/`](../../examples/humanoid-real2sim/README.md).

Run commands from the repository root. `paths.example_dir(__file__)` maps a script under `tasks/<task>/` to its corresponding example data directory, including nested tasks. Scene scripts stay beside their task commands so existing `--scene-file` and local `from scene import ...` usage continue to work.

```bash
.venv/bin/python scripts/humanoid-real2sim/kinematic_viewer.py --help
.venv/bin/python scripts/humanoid-real2sim/run.py --help
.venv/bin/python scripts/humanoid-real2sim/tasks/brush_sweep_1019/audit_sweep_cycle_v22.py
```

The last command reads the reviewed brush motion and assets under `examples/humanoid-real2sim/brush_sweep_1019/` and writes its task review result there. For reproducible commands, motion paths, and review status, read each task's example README and stage ledger.
