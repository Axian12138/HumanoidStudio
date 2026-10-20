# 1019 white waste-bin push

The user approved the revised left-hand HOI candidate for Stage 4. The left
thumb and index contact the trailing exterior of the bin, with the estimated
bin travel direction between their contact points. `stage_ledger.json` records
the source, candidate hashes, geometry audit, and current stage.

The Stage 3 kinematic viewer is available at <http://localhost:8201/>. It
shows the retargeted robot as the solid figure and the original BVH human
motion as a translucent cyan skeleton. `export_human_overlay.py` generates
`sidecars/human_motion_world.npz` from the first 9.5 seconds of the source BVH.
Only a rigid XY alignment is applied; its pelvis trajectory has 2.5 cm RMS
error against the reviewed retarget root. The human's limb lengths and heights
are preserved. Stage 4 uses the retargeted robot itself as its cyan ghost.

Stage 4 uses the ScaleBridge deployed `bfm_transformer_dataset_L_8192_64_64`
checkpoint and RSS MuJoCo Warp scene. Start it with `./run_stage4.sh`.
It starts paused at frame 0. The bin trajectory sidecar
is an estimated kinematic review aid and is **not** loaded into physics.

The 0.10 scene follows `ScaleTrack/data/1012test/trashbin-v2`: the original
25 bin collision hulls plus its collision-only mouth cylinder (94 mm radius,
24 mm height, centered 270 mm above the base), 0.3 kg mass, and low friction.
The USD has separate static/dynamic values 0.10/0.08; RSS exposes one
Coulomb coefficient per shape, set here to 0.10. The source USD variant is
copied under `assets/` for provenance. The previous 0.50-friction, 25-hull
physical baseline is preserved in `rollouts/stage4_initial/motion_repair/`.
After the operator observed tipping, `scene_mu_0p05.py` halves the RSS bin
friction to 0.05 without changing the mass or collision geometry. Its own USD
variant records static/dynamic 0.05/0.04. The 0.10 playback cache is preserved
in `rollouts/stage4_trashbin_v2/motion_repair/` (428 frames).
The next trial restores RSS friction 0.10 and lowers the local center of mass
from the scene's automatic 123.3 mm to 30 mm. It retains the 0.3 kg mass and
26 collision parts. Its USD copy records the same 30 mm center of mass and
0.10/0.08 static/dynamic friction. The 0.05 playback cache is preserved in
`rollouts/stage4_mu_0p05/motion_repair/` (469 frames).

The 0.10/low-COM trial still tipped because the floor's friction is 1.0 and
MuJoCo chose that larger value for equal-priority bin-floor contacts. The
current `scene_mu_0p10_com_0p03_contactfix.py` gives only the bin's 26
collision geoms priority 1, selecting their 0.10 contact friction while
leaving foot-floor friction alone. Its complete 474-frame physical baseline
is saved under `rollouts/stage4_mu_0p10_com_0p03_contactfix/motion_repair/`.
The bin stayed upright: maximum tilt 0.85 degrees and horizontal travel
0.836 m. See `review/stage4_contact_friction_comparison.json` for the direct
rollout comparison. It is still an unreviewed Stage 4 candidate.

## Stage 5: coffee table and bin friction

`scene.py` now points to the Stage 5 object trial. The previous default scene
is preserved as `scene_stage4_original.py`; the corrected 0.10 contact scene
and its rollout remain unchanged. `export_coffee_table.py` extracts the 56
visible mesh parts and eight authored collision parts of
`/World/MeasuredFurniture/CoffeeTable` from the IKEA 1012 room USD into
`assets/coffee_table_*.obj`. The table is world welded at its original pose.
The bin remains 0.3 kg with a local COM 30 mm above its base and the same 26
collision pieces. Its effective RSS/MuJoCo contact friction is now 0.15; the
matching USD variant `assets/white_waste_bin_mu_0p15_com_0p03_guard_v2.usd`
records static/dynamic friction 0.15/0.12. The higher-priority bin contact
geoms ensure MuJoCo uses 0.15 against the floor and table, rather than the
higher friction of those fixtures.

`stage5_replay.py` ran the same 474-frame reference and ScaleBridge tracker
with the table at friction 0.15 and 0.20, without opening another repair
editor. The 0.20 comparison also has its own USD variant under `assets/`.
Full physical rollouts are preserved under
`rollouts/stage5_coffee_mu_0p15/` and `rollouts/stage5_coffee_mu_0p20/`.
The 0.15 trial finished at (0.683, 0.468) m, 3.6 cm from the estimated bin
endpoint, with maximum 3.43° tilt. The 0.20 trial stopped earlier at
(0.650, 0.512) m, 6.2 cm from that endpoint. The top view and measurements
are in `review/stage5_coffee_mu_0p15_top.png` and the adjacent metrics JSON.
MuJoCo Warp repeatedly reported a line-search iteration warning. At the
table, the robot's thumb and index were already farther from the bin than
the desired contact, so the simultaneous hand–bin–table pinch still needs
motion/contact review before this candidate can be accepted as final data.

## Stage 4 tracker audit

The Stage 4 and Stage 5 launches use ScaleBridge's
`bfm_transformer_dataset_L_8192_64_64/model_22200_tensorrt.pt`, a different
policy from the one in the supplied `play_tracking_only.sh`. The two policy
metadata files have identical 29 joint names, 14 selected body names, stiffness,
damping, and torque limits. Both receive reference targets at offsets
`[0, 1, 2, 3, 4, 32]` in world coordinates; ScaleBridge converts them to
target-to-live-root **and** target-to-live-body features. The first feature
changes when the physical root drifts, so global position error remains visible
to the policy. The gravity convention is world negative Z and quaternions are
passed as WXYZ to the adapter.

The original ScaleTrack checkpoint was tested in the same Stage 4 scene.
It reduced some later body-position errors, but its left-wrist mean error
before bin contact (frames 200–299) grew from 3.4 cm to 10.9 cm. The bin
finished at (0.406, 0.971) m instead of (0.771, 0.379) m because the hand
did not maintain the push. The editor therefore retains the deployed
ScaleBridge checkpoint. Both Stage 4 rollouts, the Stage 5 diagnostic replay,
and body error summaries are recorded in `review/tracker_stage4_audit.json`.
The residual tracking/contact error is not explained by a missing future
reference or a root-relative-only target.

The Stage 5 dynamics review is running at <http://localhost:8193/>. It opens
paused at frame 0 with the complete 474-frame physical cache. The same view
contains the coffee table, the actual physics robot/bin, a translucent cyan
retargeted robot reference, and the synchronized original human head video.
The timeline only loads cached physical states. `--review-only` hides repair
controls and makes the page read-only.

Comparing the human video, the kinematic reference and the physical rollout
corrects the earlier table-only interpretation. The estimated final bin
center plus its 102 mm rim radius reaches x≈0.755 m, nearly the table's
x=0.760 m left edge. The hand/bin separation starts around frame 360 while
the physical bin center is still at x≈0.583 m, before reaching that edge.
At frame 375 the projected physical contact-point gap is about 9.7 cm,
versus about 1.6 cm for the reference contact points. See
`review/stage5_human_vs_physics_contact.json`. Moving the table alone cannot
restore the original human's continuous push; the tracker/contact response
needs further diagnosis.

```bash
scripts/humanoid-real2sim/tasks/push_bin_1019/run_stage4.sh
```

Control page: <http://localhost:8181/>. The completed 474-frame physical
baseline is cached; it is rewound to frame 0 for review. Press Play to replay
the cache, or click any cached frame to inspect it. New repairs branch from
the selected physical slot. The timeline then caches the robot and bin
physical state for rollback. The output remains an unreviewed candidate until
the operator has inspected the tracking, contact, and bin response.
