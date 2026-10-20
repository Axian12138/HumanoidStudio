# SPDX-License-Identifier: Apache-2.0
"""G1 hurdle traversal scene reconstructed from the supplied 2.1 s GIF.

Run with ``uv run examples/humanoid_hurdle/scene.py``. The scene is a task
environment: it does not claim to reproduce the policy shown in the reference.
"""

from __future__ import annotations

import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

import robosimstudio as rss

HERE = Path(__file__).resolve().parent
DEFAULT_G1 = HERE.parents[2] / "ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof/g1_29dof.xml"
G1_MJCF = Path(os.environ.get("G1_MJCF_PATH", DEFAULT_G1)).expanduser().resolve()
if not G1_MJCF.is_file():
    raise FileNotFoundError(f"Set G1_MJCF_PATH to a G1 MJCF with its meshes: {G1_MJCF}")


def position_control_asset(source: Path) -> tuple[tempfile.TemporaryDirectory, Path]:
    """Adapt the source's torque motors to RoboSimStudio joint targets."""
    tree = ET.parse(source)
    root = tree.getroot()
    joints = {node.get("name"): node for node in root.findall(".//worldbody//joint")}
    motors = root.findall("./actuator/motor")
    if len(motors) != 29:
        raise ValueError(f"Expected 29 G1 motors in {source}, found {len(motors)}")
    for motor in motors:
        joint = joints[motor.get("joint")]
        force_range = motor.get("ctrlrange")
        motor.tag = "position"
        motor.attrib.clear()
        motor.set("name", joint.get("name"))
        motor.set("joint", joint.get("name"))
        motor.set("kp", "120")
        motor.set("kv", "8")
        motor.set("ctrllimited", "true")
        motor.set("ctrlrange", joint.get("range"))
        motor.set("forcelimited", "true")
        motor.set("forcerange", force_range)
    temporary = tempfile.TemporaryDirectory(prefix="rss_g1_hurdle_")
    folder = Path(temporary.name)
    (folder / "meshes").symlink_to(source.parent / "meshes", target_is_directory=True)
    output = folder / "g1_position.xml"
    tree.write(output, encoding="unicode")
    return temporary, output


_asset_temp, POSITION_G1_MJCF = position_control_asset(G1_MJCF)

# World frame: +x is forward, +z is up. These are approximate dimensions from
# the rendered reference, rather than measurements recoverable from its pixels.
START_X = -0.90
HURDLE_X = 0.0
HURDLE_WIDTH = 0.065
HURDLE_LENGTH = 1.35
HURDLE_HEIGHT = 0.08
GOAL_X = 0.98


G1 = rss.EmbodimentSpec(
    name="g1_hurdle",
    asset=rss.AssetSpec(path=str(POSITION_G1_MJCF)),
    preserve_free_bodies=True,
    enable_self_collisions=False,
    arms=(
        # The joint-position action mode needs equal-width groups. One group
        # exposes all 29 motors; no Cartesian IK is used for locomotion.
        rss.ArmSpec(
            name="whole_body",
            joint_names=(
                "left_hip_pitch_joint",
                "left_hip_roll_joint",
                "left_hip_yaw_joint",
                "left_knee_joint",
                "left_ankle_pitch_joint",
                "left_ankle_roll_joint",
                "right_hip_pitch_joint",
                "right_hip_roll_joint",
                "right_hip_yaw_joint",
                "right_knee_joint",
                "right_ankle_pitch_joint",
                "right_ankle_roll_joint",
                "waist_yaw_joint",
                "waist_roll_joint",
                "waist_pitch_joint",
                "left_shoulder_pitch_joint",
                "left_shoulder_roll_joint",
                "left_shoulder_yaw_joint",
                "left_elbow_joint",
                "left_wrist_roll_joint",
                "left_wrist_pitch_joint",
                "left_wrist_yaw_joint",
                "right_shoulder_pitch_joint",
                "right_shoulder_roll_joint",
                "right_shoulder_yaw_joint",
                "right_elbow_joint",
                "right_wrist_roll_joint",
                "right_wrist_pitch_joint",
                "right_wrist_yaw_joint",
            ),
            ee_body_name="right_wrist_yaw_link",
            home_q=(
                -0.312,
                0,
                0,
                0.669,
                -0.363,
                0,
                -0.312,
                0,
                0,
                0.669,
                -0.363,
                0,
                0,
                0,
                0,
                0.2,
                0.2,
                0,
                0.6,
                0,
                0,
                0,
                0.2,
                -0.2,
                0,
                0.6,
                0,
                0,
                0,
            ),
        ),
    ),
)

scene = rss.Scene(cell="void", name="g1_hurdle_traverse", physics=rss.MUJOCO_500HZ)
scene.add(
    rss.slab(
        name="hurdle",
        half=(HURDLE_WIDTH / 2, HURDLE_LENGTH / 2, HURDLE_HEIGHT / 2),
        at=(HURDLE_X, 0, HURDLE_HEIGHT / 2),
        color=(0.48, 0.50, 0.51),
        friction=0.8,
    )
)
scene.add(
    rss.slab(
        name="goal_marker",
        half=(0.025, 0.32, 0.002),
        at=(GOAL_X, 0, 0.002),
        color=(0.19, 0.73, 0.42),
        collide=False,
    )
)
scene.add(
    rss.CameraSpec(
        name="reference_view",
        width=800,
        height=450,
        fov_y=52,
        eye=(1.55, -2.45, 1.48),
        look_at=(-0.02, 0.0, 0.61),
    )
)
robot_ref = scene.add(rss.robot(G1, at=(START_X, 0, -0.03)))


class HurdleTracker:
    """Track the task outcome across control steps."""

    def __init__(self) -> None:
        self.hit_hurdle = False
        self.fell = False
        self.last: dict[str, float | bool] = {}
        self._last_step = -1

    def update(self, sim) -> dict[str, float | bool]:
        if sim.step_count <= self._last_step:
            self.hit_hurdle = False
            self.fell = False
        self._last_step = sim.step_count
        snapshot = sim.read()
        pelvis = np.asarray(snapshot.body_q[int(robot_ref.bodies[0]), :3], dtype=float)
        if snapshot.contacts is None:
            raise RuntimeError("The hurdle task requires MuJoCo contact probing")
        self.hit_hurdle |= snapshot.contacts.in_contact(robot_ref.shapes, sim["hurdle"].shapes)
        self.fell |= bool(pelvis[2] < 0.48 or abs(pelvis[1]) > 0.8)
        reached_goal = bool(pelvis[0] >= GOAL_X and abs(pelvis[1]) <= 0.32)
        self.last = {
            "pelvis_x": float(pelvis[0]),
            "pelvis_y": float(pelvis[1]),
            "pelvis_z": float(pelvis[2]),
            "hit_hurdle": self.hit_hurdle,
            "fell_or_left_lane": self.fell,
            "reached_goal": reached_goal,
            "success": reached_goal and not self.hit_hurdle and not self.fell,
        }
        return self.last


tracker = HurdleTracker()
sim = scene.build(control_mode="joint_position")
sim.post_step_hooks.append(tracker.update)

if __name__ == "__main__":
    rss.view(sim)
