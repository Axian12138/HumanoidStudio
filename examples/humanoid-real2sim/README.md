# Humanoid real2sim in RoboSimStudio

This directory holds task descriptions, configuration, source assets, motions, review evidence, and stage ledgers. The executable task code is in [`scripts/humanoid-real2sim/tasks/`](../../scripts/humanoid-real2sim/README.md); shared algorithms and the tracker page are in [robosimstudio.humanoid_real2sim](../../robosimstudio/humanoid_real2sim/README.md). [Curated action cases](cases/README.md) and [user feedback](cases/feedback/README.md) describe reusable lessons. New routine candidates and logs belong in `outputs/humanoid-real2sim/`.

Binary example assets and motions use Git LFS through the repository's `.gitattributes`; scripts and package code use ordinary Git. After cloning, run `git lfs pull` before loading example motions or meshes.

This editor plays a ScaleTrack reference with the ScaleBridge BFM tracker while
**RoboSimStudio alone steps physics**. The ScaleBridge Python process performs
policy inference; it does not open MuJoCo or Isaac Sim. RSS handles robot,
object and deformable contact through the scene's physics arrangement. The
default scene is an empty floor with G1; supply a reconstructed scene for a
real interaction.

## Install and run

To inspect a retarget before tracker playback, use the kinematic page. It
shares the reconstructed scene and Studio 3D view, but sets each reference
`qpos` directly and offers a seekable source-frame progress bar:

```bash
.venv/bin/python scripts/humanoid-real2sim/kinematic_viewer.py \
  --motion /path/to/motion.npz \
  --scene-file /path/to/reconstructed_rss_scene.py \
  --start-frame 300 --stop-frame 600 --port 8090
```

Open `http://localhost:8091/`. The range is `[start-frame, stop-frame)`;
the example reviews 6–12 seconds at 50 Hz. An optional `--object-track`
NPZ with `position_w` and `quaternion_xyzw`, plus `--object-name`, displays
estimated object poses. No ScaleBridge worker, tracker inference, physics
steps, contact validation, repair tools, or candidate export run on this page.

For the 1012 brush pickup candidate, the 6–12 second window is:

```bash
.venv/bin/python scripts/humanoid-real2sim/kinematic_viewer.py \
  --motion examples/humanoid-real2sim/brush_pick_1012/retarget/handle_kinematic_v1/motion.npz \
  --scene-file scripts/humanoid-real2sim/tasks/brush_pick_1012/scene.py \
  --object-track examples/humanoid-real2sim/brush_pick_1012/retarget/handle_kinematic_v1/object_estimate.npz \
  --object-name green_hand_brush --start-frame 300 --stop-frame 600 --port 8090
```

## Tracker repair

The RSS checkout uses its own Python 3.12 `uv` environment. The existing
ScaleBridge Python 3.11 environment loads the supplied Torch-TensorRT policy.
Both environments must be available; this particular checkpoint requires a
CUDA GPU. From the RSS repository root:

```bash
uv run --no-sync scripts/humanoid-real2sim/run.py
```

The existing repository `.venv` is sufficient for the RSS side. On a fresh
checkout, run `uv sync` first; the optional renderer download is large. The
ScaleBridge worker still uses its separate environment and needs CUDA.

To load the estimated hurdle from the supplied GIF into the repair editor:

```bash
uv run --no-sync scripts/humanoid-real2sim/run.py \
  --scene-file scripts/humanoid-real2sim/hurdle_scene.py \
  --motion /path/to/retargeted_hurdle_motion.npz
```

The hurdle is placed 0.90 m ahead of the reference's initial pelvis and is an
estimate; align it to the reconstructed clip before treating contact as data.

The default `cartwheel.npz` is 29 body joints and has no hand motion. For a
43-joint Dex3 retargeted clip, specify its motion and choose reviewed frames
that define **open** and **closed** hand poses:

```bash
uv run --no-sync scripts/humanoid-real2sim/run.py \
  --motion /path/to/motion.npz \
  --hand-open-frame 12 --hand-close-frame 70 \
  --scene-file /path/to/reconstructed_rss_scene.py
```

The open/closed frames are examples, not calibrated defaults. Inspect the
actual clip and hand geometry before choosing them. The seven-field ScaleTrack
NPZ has no joint/body names; the launcher uses
`ScaleTrack/retarget3/assets/robot/motion_order.json` and validates dimensions.
If the source uses another order, pass `--motion-order` for that exact order.

`--scene-file` must define `build_sim(asset, motion) -> rss.Sim`. `asset` is the
temporary position-actuated G1 Dex3 MJCF; `motion` contains the reference. The
factory can import Blender-derived meshes or USD and can choose an RSS physics
arrangement. It should return an already built scene with a floating G1 named
as in the supplied model. The launcher aligns its reset pose to the first
reference frame. An example of the minimum contract:

```python
def build_sim(asset, motion):
    import robosimstudio as rss
    # Add the G1 embodiment using `asset`, reconstructed geometry and cameras.
    # Use RSS physics arrangements; do not launch another physics simulator.
    scene = ...
    return scene.build(control_mode="joint_position")
```

For a solver demonstration on the default floor, `--soft` adds a sponge. Its
default Newton VBD route builds without extras but does not establish reliable
G1 hand/foam contact. `--soft --soft-solver superdex` selects RSS's optional
SuperDex route; install it with `uv sync --extra superdex`. Select and calibrate
the solver for the reconstructed object before accepting contact results.

## Editing controls

The page starts paused at the initial state. **Play** runs tracker inference
and RSS robot/object dynamics at 50 Hz. The white part of the timeline is a
contiguous cache of simulated states; gray slots have no active cache. Dragging
the timeline loads a saved robot/object state directly, without physics steps.
Playing through a cached span loads those states in order. A new repair at the
current slot invalidates only its later cache, leaving the continuous prefix
and earlier repair markers intact. Each repair start appears as an orange dot.

The page uses one **Play / Pause** button. Five scene dots mark the pelvis,
wrists, and ankles; the five labels in the control card select the same links.
Double-click a dot or label to toggle its direction arrow and XYZ translation
gizmo. Drag its tip to set direction without changing the camera pose. Each
link has its own speed, initially 0.30 m/s. Edits are staged while paused;
Play branches the physics cache only when the command changes. Rewinding and
replaying an unchanged segment reuses the cached states without another marker.
A second link can be edited in the same segment. Untouched links follow their
moving reference, and closing a repaired link's arrow holds its last target.
For a moving link, every VR-5 inference receives a future target ramp matching
the selected velocity at offsets 0, 1, 2, 3, 4 and 32. The direction is thus
visible on the first inference after Play, before a physical step has elapsed;
every future slot remains inside the repair's total displacement budget.

**Find recovery frame** opens a separate reference frame slider and overlays a
cyan 14-body reference skeleton on the current physical robot. The operator
chooses the frame that best matches the current state; Play starts a smooth
transition toward that frame and its following reference trajectory. Recovery
continues until position, orientation, and joint checks pass. Left and right
Dex3 hands each have a continuous one-dimensional open/close slider. Their
values are cached with each state and return to the reference schedule after
recovery.

Each cached slot stores joint, rigid-body, and particle positions and velocities,
the commanded target, tracker action, reference clock, and repair modes. Scene
state is written to both physics buffers when a slot is loaded. At a repair
branch, only solver warm-start/contact buffers are cleared. MuJoCo's default
reset would also reset qpos and caused the robot to jump on the first physics
step after a cached rewind.

Studio's built-in **Reset** returns to the initial state; it pauses playback
and clears the current repair branch.

## Data contract

`Save candidate rollout` writes `rollout.npz`: source-frame mapping, mode,
commanded joint targets, tracker action, and achieved joint/body/particle
states. With a 43-joint clip and completed recovery, it also writes a
seven-field `candidate_motion.npz` and `candidate_mapping.npz`. Unedited spans
come from the original motion; takeover/recovery spans use achieved full-body
states. Narrow boundary windows blend the two before MuJoCo **forward
kinematics only** computes all 50 body poses. The candidate is explicitly
unreviewed: contact, foot slip, velocity, visual alignment and a fresh RSS
rollout must be checked before training-data promotion.

No Isaac Sim physics or PhysX is invoked. RSS rendering uses its configured
renderer; the camera path is independent of the RSS physics solver.

