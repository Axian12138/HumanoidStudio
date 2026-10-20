# G1 hurdle traversal

This scene uses RoboSimStudio's Newton/MuJoCo physics to place a floating-base
G1 on level ground, facing a fixed, collidable beam. The reference GIF shows
the robot approaching, stepping over the beam, and continuing toward a goal.
The robot and beam dimensions cannot be recovered precisely from the rendered
GIF, so this scene uses a 65 mm wide, 80 mm high, 1.35 m long beam; the G1
starts 0.90 m before it and the goal is 0.98 m beyond it.

The scene supplies the robot, collision geometry, a 30D joint-position action
(29 joints plus an unused gripper slot), and a `tracker.last` status dictionary.
A run succeeds when the pelvis reaches the goal lane without any recorded beam
contact or a fall. A post-step hook checks every control step. Ground
contact is expected; "collision free" refers to the hurdle. The scene does
**not** include the controller or motion from the GIF.

The example makes a temporary MJCF with position servos from the source G1's
torque motor model. It keeps the same joints, inertias, visuals, and collision
geometries, and limits each servo by the source motor's effort range.

## Setup

From the repository root:

```bash
UV_CACHE_DIR=/home/galbot/workspace/.uv-cache uv sync --no-group rtx
uv run --no-group rtx examples/humanoid_hurdle/smoke.py
G1_MJCF_PATH=/path/to/g1_29dof.xml uv run --no-group rtx examples/humanoid_hurdle/scene.py
```

The MJCF must retain its sibling `meshes/` directory. In this workspace, the
default path points to the G1 asset in `../ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof/`.
The RoboSimStudio live page can load this file with:

```bash
cd examples/live
./ctl up --scene ../humanoid_hurdle/scene.py
```

Physics builds and steps on CPU here. This workspace has no usable NVIDIA
driver inside the restricted shell, but CPU rendering works. Run either preview:

```bash
uv run --no-group rtx examples/humanoid_hurdle/render_preview.py
MUJOCO_GL=egl uv run --no-group rtx examples/humanoid_hurdle/render_mujoco.py
```

`render_preview.py` uses RoboSimStudio's colliders-only tiled renderer.
`render_mujoco.py` uses the same G1, beam, and goal dimensions with MuJoCo's
visual meshes and needs a working EGL GPU. Both write PNGs beside the scripts.
