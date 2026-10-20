# DexAgent-inspired contact-method trial on the G1 Dex3 brush pickup

**Review status: rejected for training data.** This is a completed hand-only
experiment, not an accepted pickup. The original human-derived body prior,
source recording, and earlier candidates were not overwritten.

## What was transferred

DexAgent checkout `e29bdad8ed3e95b2c9ee2eea9044a6b4115ec633` provides
`pose_from_points`, `optimize_fingertip_contacts`, and grasp stress verifiers.
Those tools assume its Sharpa/Shadow hand rig: a virtual six-joint wrist,
named fingertip sites, and its own actuator and collision conventions. The
G1 Dex3 hand does not satisfy that interface. `dexagent_hand_only.py` uses
manually chosen brush contact points, then implements a related multi-seed
wrist/finger IK and collision refinement on the actual Dex3 FK and collision
meshes. It does not call DexAgent's original grasp tools, execute its agent
workflow, or use BODex/DexGraspNet model weights. This experiment cannot
establish that DexAgent itself outperforms the existing retarget.

The human-derived 1750-frame motion supplies contact time, the approach
window and the index/middle positions along the handle. The brush's reviewed
collision hull supplies the contact surfaces. Only a virtual right wrist
SE(3) and seven right-hand joints are optimized. `hand_targets.npz` stores
the hand target over frames 450–660; `review.json` gives the grasp frame,
contact points and geometry measurements.

`project_dexagent_hand.py` maps that hand target to the existing seven-joint
right arm solely so the ScaleBridge tracker can be tested. The root, legs,
torso and left arm retain the prior. `projected_motion.npz` is a validation
input, not a new full-body retarget method.

## Evidence

- Static target: contact-frame right-wrist translation is about 42 mm upward.
  Keyframe arm IK position residual stays below 0.05 mm and orientation
  residual below 0.003 rad. The largest finger-mesh penetration into the
  brush grip is 1.43 mm. These are kinematic checks.
- Actual ScaleBridge TensorRT tracker plus RSS MuJoCo Warp physics:
  `physical_replay/rollout.npz` contains 610 executed frames. At frame 560
  the wrist is 108 mm from its target and the pelvis 99 mm from its target.
  The seven finger joints track their commands to about 0.03–0.05 rad, so
  finger actuation is not the dominant observed error.
- The brush moves about 115 mm sideways from frames 525 to 599 and ends
  41 mm lower than at frame 500. It never passes the 50 mm lift gate.
  `acceptance.json` has exact readings and hashes; `acceptance.png` plots
  brush height and wrist/pelvis tracking error.

Thus the static hand contact solution did not survive whole-body tracking.
The next controlled experiment needs a root/wrist/feet target that can make
the *achieved* hand arrive at the handle while preserving stance. This is the
motivation for the proposed five-point retarget mode; the hand-only result
must not be promoted to ScaleTrack training data.

## Reproduce

From the repository root:

```bash
.venv/bin/python scripts/humanoid-real2sim/tasks/brush_pick_1012/dexagent_hand_only.py
.venv/bin/python scripts/humanoid-real2sim/tasks/brush_pick_1012/project_dexagent_hand.py
WARP_CACHE_PATH=/tmp/warp_brush_dexagent .venv/bin/python \
  scripts/humanoid-real2sim/tasks/brush_pick_1012/probe_dexagent_motion.py \
  --motion examples/humanoid-real2sim/brush_pick_1012/retarget/dexagent_hand_only_v1/projected_motion.npz \
  --out examples/humanoid-real2sim/brush_pick_1012/retarget/dexagent_hand_only_v1/physical_replay \
  --frames 610
.venv/bin/python scripts/humanoid-real2sim/tasks/brush_pick_1012/review_dexagent.py
```

The replay requires a CUDA GPU and the existing ScaleBridge TensorRT
checkpoint. `probe_dexagent_motion.py --frames 0` rebuilds
`physical_replay/scene_labels.json` without overwriting the saved rollout.
