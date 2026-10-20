# 1012 brush pickup, source 6–12 s

The operator approved the `palm_waist_v3` kinematic playback on 2026-10-19 as the stage-4 input. This approval covers the viewed approach and whole-body pose only; tracker execution and physical brush pickup remain to be checked in the RSS editor.

Stage 4 opens paused at source frame 300 and stops playback at frame 600:

```bash
.venv/bin/python scripts/humanoid-real2sim/run.py \
  --motion examples/humanoid-real2sim/brush_pick_1012/retarget/palm_waist_v3/motion.npz \
  --scene-file scripts/humanoid-real2sim/tasks/brush_pick_1012/scene.py \
  --start-frame 300 --stop-frame 600 \
  --reference-pid-wrists right \
  --hand-open-frame 490 --hand-close-frame 560 \
  --hand-kp 20 --hand-kv 0.2 \
  --port 8120 --out examples/humanoid-real2sim/brush_pick_1012/rollouts/palm_waist_v3_stage4_6d_grip
```

The control page is `http://localhost:8121/`. Press Play to run the real ScaleBridge tracker and RSS object physics; the kinematic object sidecar is not loaded into this stage. Both hand sliders now run from an explicit open pose to near-limit flexion targets with `kp=20`, `kv=0.2`. Each repaired link has a fixed 6D target and separate linear/angular speed caps; the translucent local hand/foot model previews its target pose. The future target setpoint accumulates every frame even when the physical link lags, while measured pose determines the remaining direction and stop condition. The repaired wrist is controlled by tracker rather than its ordinary reference PID during takeover.
During recovery, both hands retain their entry control targets while the body interpolates. A manual hand slider setting remains active after recovery until the operator changes it. The exported candidate uses achieved finger joints for held hands, so it does not silently replace a physical grasp with source hand angles.

The previous 349-frame branch was saved at `rollouts/palm_waist_v3_stage4_6d_grip/motion_repair/rollout.npz`. The corrected recovery review page is `http://localhost:8131/`: it restores the first 249 physical slots through source frame 548, with the right wrist held at its achieved pose and right-hand closure at 100%. It is paused in recovery selection at source frame 549 for a 50-frame replay. The original saved rollout remains available for comparison.
The right wrist's roll, pitch and yaw position targets come directly from the selected reference motion and are applied through the RSS position actuators. The other controlled joints still use ScaleBridge tracker targets. The wrist's achieved angles and brush response need physical review.
An initial 13-frame RSS playback from source frames 300–312 verified that each commanded right-wrist target matched refmotion exactly (maximum difference 0 rad). Achieved mean absolute roll/pitch/yaw errors were 0.0041/0.0242/0.0084 rad over this short interval. This is a control-path check, not a pickup result. The page was reset to frame 300 afterward.

The subsequent grip diagnosis is in `review/stage4_grip_diagnosis.json`. The apparent brush lift in an unedited wrist-PID rollout was a collision-driven separation, not a held grasp: wrist-to-grip distance increased from 145 mm at frame 560 to 429 mm at frame 599. The old experiments used ScaleBridge's `g1_29dof_dex3.xml`, whose wrist-to-finger geometry differs from the approved `g1_29dof_dex3_offsethand.usda`; they are historical evidence only. Stage 4 and the kinematic viewer now load the exact `g1_29dof_dex3_offsethand.xml` used by the retarget FK and referenced by that USD. Isolated tests on this unified model show substantial finger target lag at the default position gain and no measured finger–brush force in the uncorrected pose. `retarget/palm_waist_v3_offsethand_clamp/motion.npz` is a separate 2 mm mesh-interference closure candidate. With a high position gain of 20, it produced thumb-tip and index-tip contact peaks of 26.8 N and 43.3 N, but the brush shifted about 44 mm laterally and was not held; the candidate is **not** a successful grasp. Handle alignment remains for operator repair, and lift retention remains unverified.

This directory is an independent stage-3 candidate for recording
`1-2026-10-12-17-32-36`. It reuses the reviewed stage8 full-body prior and
the original IKEA world frame. It does not change the ScaleTrack inputs.

## Available result

- `retarget/palm_waist_v3/motion.npz`: latest candidate. It keeps the palm-first path and allows at most 0.2 rad of waist yaw/roll/pitch adjustment during the interaction, leaving the root and legs on the existing body prior. This reduces the remaining wrist pitch and distributes the orientation change across the torso and right arm. Review it before choosing between this and v2; neither has a tracker or physics grasp validation.
- `retarget/palm_approach_v2/motion.npz`: revised 6–12 s candidate after user visual review. The open right hand moves outside the near side of the basket, then enters laterally toward the exposed handle with its palm facing the motion. Wrist orientation is regularized throughout the approach rather than only beyond a 65° threshold. The scene and fixed hand timing are unchanged. `object_estimate.npz` remains an estimated held path, not a physics result.
- `retarget/handle_kinematic_v1/motion.npz`: new 6–12 s right-hand handle pickup reference. It reuses the existing global/stance body prior, solves the right arm toward a handle grasp pose, and assigns a fixed Dex3 closing schedule. The full 1750-frame motion keeps original source indexing; arm corrections end by frame 625 and finger commands rejoin the prior by frame 660.
- `retarget/handle_kinematic_v1/review.json`, `audit.json`, `review_closeup.png`, `review_fullbody.png`: handle and finger geometry, source speed limits, basket-hull vertex sweep, top/side close-ups, and same-frame human/prior/candidate body overlays. `object_estimate.npz` is a kinematic held-brush path after frame 560, not a measured physics trajectory.
- `retarget/kinematic_motion.npz`: 1750 frames at 50 Hz, seven-field 43-joint / 50-body reference. Keeping the full motion preserves source-frame indexing; the scoped review interval is frames `[300,600)` (6–12 s).
- `retarget/manifest.json`: source hash and the fixed Dex3 command times. Only right-hand finger commands in frames 445–560 differ from stage8 joint positions. Root, legs, torso, left arm and right-arm joint positions remain on the stage8 prior. FK and velocity fields were regenerated.
- `scene.py`: RSS `build_sim(asset, motion)` with the G1 floating root, table support proxy, white storage basket and free green brush. Blender visual meshes and the source-authored collision hulls are under `assets/`. The support slab, masses and friction are provisional; the scene is a local interaction region, not the full living room.
- `review/rest_selected.json`: five seconds of free brush/basket RSS MuJoCo Warp dynamics from the selected pose. Brush moved 1.54 mm and rotated 0.41° overall; during the last second each position coordinate varied by at most 0.43 mm. It stayed in the basket without toppling. Solver line-search warnings and residual contact jitter remain. This is evidence of support, not grasp success.
- Archived physical experiments under `retarget/dynamic_v1/`, `stage8_timing_probe/`, `stage4_wrist_probe/`, `grasp_geometry_v1/`, and `root_catchup_v1/`. The optimizer and its configurations have been removed; these files remain as failed-pickup evidence.
- `review/*_analysis.json`: achieved brush height, movement, and hand-link proximity from each final replay.

The selected brush pose in IKEA world is position `(0.871603, -0.587127, 0.438703)` m and quaternion `xyzw=(0.002078, 0.143548, -0.008530, 0.989604)`. The brush head is near `(0.9453, -0.5884, 0.4586)` m inside the white basket; the grip midpoint is near `(0.8120, -0.5862, 0.5190)` m, about 5 cm outside the basket's near-side footprint. This placement came from RSS free settling of a 13 mm basketward adjustment and leaves the handle exposed. It differs from the stage8 kinematic object sidecar, so physical pickup alignment and surrounding collision clearance require the dynamic replay and close-up review.

## Fixed hand plan

The earlier prepared reference starts opening Right Dex3 at 8.90 s, reaches the open target by 9.80 s, stays open through 10.40 s, then closes through 11.20 s. First surface contact is estimated at 10.50 s with ±0.30 s source uncertainty. These seven finger joint trajectories are fixed in the kinematic reference. Actual opening, first contact, closure, lift and slip require RSS physical playback. The previous stage8 sampled geometry audit saw shallow hand/basket intersections near 10.18–10.28 s; the new candidate has its own sweep audit.

The new handle candidate starts closing at 10.36 s (frame 518) and completes at 11.20 s (frame 560); the grasp prior begins at frame 522. Its closed finger angles were fitted once against the actual brush grip collision hull, then held as a fixed command trajectory during arm retarget. The selected grip is 15 mm farther out along the same stable brush handle than the stage4 grip point. The brush itself was not moved.

## Handle retarget result

The earlier reference had a 64.5 mm fist-eye error at frame 525 against this brush placement. The new candidate has 0.048 mm error there; over frames 522–545 the mean/max fist-eye errors are below 0.1/0.3 mm, and mean/max handle-axis errors are below 0.1/0.3 degrees. The three intended distal finger surfaces have about 0.4 mm geometric clearance from the grip hull at closure. The changed joints remain below 0.89 of their URDF speed limits; root, legs, torso, and left arm are identical to the prior. The sampled hand/basket vertex audit found no intersections over frames 450–600. The estimated held-brush path has no basket-hull vertices deeper than 1 mm over frames 560–600.

This is an **unaccepted kinematic candidate**. The close-up plot shows mesh vertices and collision proxies, not a camera-matched render. The vertex audit does not rule out edge-only or between-frame intersections. Finger clearance does not establish grasp force, actual motor closure, stable lift, or tracker following; those require the separate RSS playback and physical review in stages 4/5.

## Palm-first revision after visual review

The first candidate aligned the fist eye and brush axis but never constrained the palm or the direction of entry. It descended from above and flipped its hand while already near the handle. The revised candidate makes frames 510–522 a near-side lateral entry; the palm-forward dot product with the eye's travel direction has mean 0.964 and minimum 0.891. At frames 522–545, the fist-eye error is 2.01 mm mean / 2.18 mm max and handle-axis error is 6.32° mean / 6.69° max. The 470–560 frame three-axis wrist-angle norm falls from 1.56 to 1.18 rad mean and from 2.00 to 1.58 rad maximum. The right wrist pitch still reaches about 1.35 rad; this is an unresolved tracker-feasibility risk with the inherited root/stance and fixed brush pose. The revision does not claim a validated grasp.

The sampled hand/basket hull audit found zero inserted vertices in frames 450–600. The estimated held brush has zero hull vertices deeper than 1 mm in the basket during frames 560–600. Hand/brush mesh samples stayed separated through frame 525; the fitted closed fingers are about 0.4 mm off the grip hull at frame 560. These sampled checks do not establish continuous collision clearance or contact force.

The follow-up v3 candidate lowers the wrist three-axis norm further to 1.05 rad mean / 1.34 rad maximum over frames 470–560. At frame 525, wrist roll/pitch/yaw are about 0.90/0.87/0.47 rad. Palm-forward alignment over frames 510–521 is 0.978 mean / 0.957 minimum. Contact fist-eye error is 3.19 mm mean / 6.83 mm maximum, and handle-axis error is 10.78° mean. Waist changes reach 0.2 rad; the left wrist shifts up to 6.0 cm even though its own joints, root and legs remain unchanged. This full-body deviation requires visual review. The sampled hand/basket and estimated brush/basket hull audits found no penetration, and changed-joint speeds stay below 0.89 of the URDF limits. Smoothing reduced the maximum edited-joint acceleration to about 100 rad/s². A kinematic palm-first pose still does not establish actual grasp force or trackability.

To inspect this version with the 3.5 kinematic player:

```bash
.venv/bin/python scripts/humanoid-real2sim/kinematic_viewer.py \
  --motion examples/humanoid-real2sim/brush_pick_1012/retarget/palm_approach_v2/motion.npz \
  --scene-file scripts/humanoid-real2sim/tasks/brush_pick_1012/scene.py \
  --object-track examples/humanoid-real2sim/brush_pick_1012/retarget/palm_approach_v2/object_estimate.npz \
  --object-name green_hand_brush --start-frame 300 --stop-frame 600 --port 8092
```

For the waist-assisted candidate, replace both `palm_approach_v2` paths above with `palm_waist_v3`.

## Reproduce

From the RSS repository root, using its `.venv`:

```bash
WARP_CACHE_PATH=/tmp/warp_brush_pick .venv/bin/python scripts/humanoid-real2sim/tasks/brush_pick_1012/prepare_kinematic.py
WARP_CACHE_PATH=/tmp/warp_brush_pick .venv/bin/python scripts/humanoid-real2sim/tasks/brush_pick_1012/rest_check.py --out examples/humanoid-real2sim/brush_pick_1012/review/rest_selected.json
.venv/bin/python scripts/humanoid-real2sim/tasks/brush_pick_1012/retarget_handle.py
.venv/bin/python scripts/humanoid-real2sim/tasks/brush_pick_1012/audit_handle.py
.venv/bin/python scripts/humanoid-real2sim/tasks/brush_pick_1012/review_handle.py
.venv/bin/python scripts/humanoid-real2sim/tasks/brush_pick_1012/review_fullbody.py
.venv/bin/python scripts/humanoid-real2sim/tasks/brush_pick_1012/retarget_palm_approach.py
.venv/bin/python scripts/humanoid-real2sim/tasks/brush_pick_1012/audit_palm_approach.py
.venv/bin/python scripts/humanoid-real2sim/tasks/brush_pick_1012/review_palm_approach.py
.venv/bin/python scripts/humanoid-real2sim/tasks/brush_pick_1012/review_palm_fullbody.py
.venv/bin/python scripts/humanoid-real2sim/tasks/brush_pick_1012/retarget_palm_waist.py
.venv/bin/python scripts/humanoid-real2sim/tasks/brush_pick_1012/audit_palm_waist.py
.venv/bin/python scripts/humanoid-real2sim/tasks/brush_pick_1012/review_palm_waist.py
.venv/bin/python scripts/humanoid-real2sim/tasks/brush_pick_1012/review_palm_waist_fullbody.py
```

## Archived physical result, 2026-10-19

The selected stage8-based dynamic search used four right-arm joints as variables, two control points each, ten evaluated references, two baseline repeats, and a separate final replay. The best reference retained the baseline: its apparent improvements did not exceed repeated-rollout variation. At 12 s the brush was 2.2 mm below its 10 s height; maximum later rise was 9.6 mm, with no sustained 5 cm lift. The right thumb distal link origin remained at least 153 mm from the estimated handle grasp point during 10–11 s, while the middle-finger distal link came within 33 mm. Link-origin distances are a diagnostic, not collision-surface distances. The rollout format does not expose contact force or achieved Dex3 joint angles, so physical closure force and slip cannot yet be measured.

Using the original stage8 earlier closing schedule also failed to lift the brush. The stage4 bent-wrist prior pushed it about 11 cm sideways and raised it only 25 mm by 12 s; it did not hold the brush. Gravity trials with larger brush-handle pitch (25–45°) rolled or threw the brush from the basket, so those poses were rejected. The selected stable placement is unchanged.

The candidate remains **unaccepted**. The next retarget must address the achieved thumb/handle geometry and tracker lag, then repeat the free-object pickup test. A lower cost or motion-only overlay is not sufficient for acceptance. The archived results used fixed hand commands and object physics.

The archived grasp-geometry search varied right wrist roll/pitch/yaw, shoulder pitch and elbow with bounded reference corrections. No candidate improved beyond repeated-baseline variation; its independent replay retained the baseline, with 4.6 mm brush rise at 12 s and no sustained lift. Actual thumb-link origin stayed at least 153 mm from the estimated grip point. This identifies a substantial hand pose / tracker execution mismatch requiring a new whole-arm contact reference. Moderate or large upward-handle placement trials were physically rejected: a 25° trial rolled about 60° in the basket, while 30–45° trials fell or launched out. Their free-rest traces are under `review/rest_p*.json`.

At the grasp frame 10.3 s, the achieved pelvis is about 113 mm behind the source pelvis; the achieved wrist is about 119 mm behind its reference. The root-correction search reduced its weighted task cost from repeated baselines 37.54/37.26 to 32.62 on independent replay, but the brush stayed in the basket. Its selected x correction was negative, illustrating that the current weighted cost can reward avoiding a bad push without completing the pick. This is not an accepted dynamic reference.

An anticipatory time-warp experiment also remains rejected. Advancing the 8.0–10.5 s body approach by 5, 10, or 25 frames raises the already near-limit right-knee speed from 5.999 rad/s to 6.097, 6.194, or 7.105 rad/s. These motions were **not** sent to the tracker. A new kinematic whole-body retarget needs coordinated feet/stance anchors and a reviewed contact pose before physical pickup playback.
