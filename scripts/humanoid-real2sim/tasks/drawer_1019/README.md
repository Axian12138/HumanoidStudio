# Drawer HOI finetune

This task refines the existing `05_drawer_event_timing/v5` composite in `scoped_candidate_v3`; it is not a new body-prior retarget. The source window is 49.5–55.5s, including approach and withdrawal. Source motion, human, scene and drawer timing remain explicitly versioned.

Create a run with `robosimstudio.humanoid_real2sim.paths.run_dir("sequence_1019/05_drawer_hoi")`, then run:

```bash
.venv/bin/python scripts/humanoid-real2sim/tasks/drawer_1019/prepare.py --run "$run"
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/humanoid-real2sim/tasks/drawer_1019/refine.py --run "$run" --version v2
bash scripts/humanoid-real2sim/tasks/drawer_1019/view.sh "$run" v2
```

`prepare.py` uses the latest hat run recorded by the preceding task in `/tmp/hat_natural_run_path`; its resolved source paths are saved in the generated scene config. The drawer motion remains an estimated kinematic trajectory, not a measured force-driven rollout.

Independent checks use `audit_joint_contract.py`, `audit_chair_brace_connection.py` with the registered room and source offset 49.5, `hat_cylinder_1019/audit_self.py` with `--source-start-s 49.5`, and `03_04_05_joint/audit_native_drawer_hand.py` with `--source-start-s 49.5`. For the latter expose `scene_tracks/drawer_slide.npz` as candidate `drawer_slide.npz` and use the exact XML in the scene config. Limit BLAS to one thread for geometric queries. Do not declare optimization convergence solely because its evaluation budget ends.
