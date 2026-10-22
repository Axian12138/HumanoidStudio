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
