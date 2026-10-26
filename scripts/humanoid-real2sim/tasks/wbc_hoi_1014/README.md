# Review paired human and NX1 teleoperation motion

These adapters reuse `KinematicStudio`; they do not retarget to G1 or advance physics. `prepare.py` reads original `traj.npz`, uses the recorded 32-joint order, and aligns original `axis_skel_pos` into the scene with the fixed transform between `qpos_ref` and `qpos`. It preserves the original 50 Hz frames. The dataset's extra padding and Dex3 arrays are not NX1 joint coordinates.

The run directory contains original episodes, the specified NX1 description checkout, selected Room_assets USD sources, adaptation outputs, and audits. Use the v1.0.1 URDF, not the repository's latest robot variant. The source capture's simplified MJCF is unavailable; the 33 original visual mesh hashes match the recording metadata. USD scene object transforms are compared against the recorded `scene_config.json`.

```bash
RUN=outputs/humanoid-real2sim/2026-10-22/wbc-hoi-1014/runs/14-02-09.180626+0800
.venv/bin/python scripts/humanoid-real2sim/tasks/wbc_hoi_1014/prepare.py "$RUN"
PYTHONPATH=/tmp/wbc-usd .venv/bin/python scripts/humanoid-real2sim/tasks/wbc_hoi_1014/export_scene.py "$RUN"
.venv/bin/python scripts/humanoid-real2sim/tasks/wbc_hoi_1014/view.py "$RUN" --port 8510
```

`export_scene.py` requires `usd-core` (installed separately under `/tmp/wbc-usd` in this run). Open only the timeline page at http://localhost:8511/. The sample dropdown jumps between episodes. Episode 02 is displayed 12 m along X, including its human, robot, room, and fruit marker. The timeline concatenates the episodes: frame 1031 is episode 02's source frame 0. This is a review clock, not one continuous recording or motion. `segments.json` records the boundaries.

Rooms use original geometry and flat diffuse colors without texture maps. Missing fruit meshes are explicitly represented by orange position markers, not substitute fruit geometry. Original recorded object poses remain available in the sidecars. The screenshot's contacts cannot establish physical grasping or tracking success.

## Human → G1 Stage 3 candidate

`retarget_g1.py` consumes one original `traj.npz` and its sibling `motion.bvh`.
It uses the paired source roots only for the fixed ref-to-world calibration;
G1 joint targets come from the human skeleton. Never feed the combined review
motion, the second room's +12 m display transform, or source NX1 joint angles
into this solver.

The original files declare 50 Hz, but their frame intervals are not uniform.
For episode `20261008_172423_003`, `t_wall` spans 25.2932 s and agrees with
metadata `record_duration_s`; `t_sim` spans only 8.57 s. The adapter explicitly
resamples the wall clock to 50 Hz (1265 frames, 25.28 s), preserving both original
clocks and fractional source-frame indices in `sidecars/timing.npz`. The original
NX1 viewer above used the nominal frame clock; it is not a timing audit.

Use ScaleTrack's Python 3.11 environment for retarget3. The RSS viewer uses this
repository's Python 3.12 environment. Outputs must be a fresh candidate directory
inside `robosimstudio.humanoid_real2sim.paths.run_dir('wbc-hoi-1014')`:

```bash
PYTHONPATH=/home/galbot/workspace/ScaleTrack/retarget3/src:/home/galbot/workspace/robosimstudio \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/home/galbot/workspace/ScaleTrack/.venv/bin/python \
  scripts/humanoid-real2sim/tasks/wbc_hoi_1014/retarget_g1.py \
  /path/to/original/01_1014/traj.npz --output /path/to/run/retarget/body_v1 \
  --fists /home/galbot/workspace/ScaleTrack/data/1012test/ground_stage2/fists.json
```

The solver uses the actual 43-joint G1+Dex3 model and fixed neutral fists. It
preserves world XY and stance anchors. BVH supplies foot heading; its fixed
bone offsets differ from `axis_skel_pos` by up to 29.4 mm in this episode, while
the roots agree within 2 µm. Original position targets are retained. The adapter
records this discrepancy rather than claiming an exact BVH skeleton match.

The reusable `retarget/window_prior.py` solves 60-frame windows, commits 20
frames at a time, and enforces stance, existing geometric guards and velocity
bounds across seams. The original greedy solver fails at landing even though
the pose is statically feasible; neither foot sliding nor a speed-limit increase
is an acceptable workaround. This window solver does not yet optimize furniture
collisions. Use `export_collision.py` and `audit_g1.py --room-triangles ...` for
an independent framewise room check; that check is not full continuous collision
detection or physics.

`view_g1.py` reuses `KinematicStudio`, with original human, actual G1 and the
original room. It takes `--source-room`, `--robot`, `--motion-order` and `--port`.
Only its timeline URL (port + 1) is a review entry. Missing fruit geometry is
shown as an explicitly labeled position marker and cannot support HOI contact
acceptance. Body prior, HOI, human review and physical acceptance remain separate
states in each run's stage ledger.
