# Humanoid real2sim commands

This directory contains command line entry points and task specific programs.
Shared runtime code belongs in [`robosimstudio/humanoid_real2sim/`](../../robosimstudio/humanoid_real2sim/README.md). A run's full experimental results belong in `outputs/humanoid-real2sim/<date>/<task>/`, with date first; distilled action descriptions belong in [`reference-knowledge/humanoid-real2sim/`](../../reference-knowledge/humanoid-real2sim/README.md), while task assets, motions and review data stay in outputs. Examples are organized by reusable action and failure mechanism; each indexed case links a small, hashed motion/image/audit artifact and identifies successful, failed, and pending claims. Use `robosimstudio.humanoid_real2sim.paths.run_dir()` for a new run.

Run commands from the repository root. For historical scripts, `paths.example_dir(__file__)` maps a script under `tasks/<task>/` to its moved `outputs/humanoid-real2sim/2026-10-20/<task>/curated_snapshot/`, including nested tasks. This compatibility helper can still write into that local snapshot; new experiments should call `run_dir()` for an isolated run. Scene scripts stay beside their task commands so existing `--scene-file` and local `from scene import ...` usage continue to work.

The task programs reproduce versioned, reviewed examples and are excluded from the repository-wide Ruff lint and format gate to avoid rewriting their recorded implementations. Shared package code and current CLI entry points remain checked; new reusable behavior belongs there.

```bash
.venv/bin/python scripts/humanoid-real2sim/kinematic_viewer.py --help
.venv/bin/python scripts/humanoid-real2sim/run.py --help
.venv/bin/python scripts/humanoid-real2sim/tasks/brush_sweep_1019/audit_sweep_cycle_v22.py
```

The last command reads the reviewed brush motion and assets under `outputs/humanoid-real2sim/2026-10-20/brush_sweep_1019/curated_snapshot/` and writes its task review result there. For reproducible commands, motion paths, and review status, read each task's example README and stage ledger.
