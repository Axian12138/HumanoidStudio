"""Example reconstructed hurdle for ``run.py --scene-file``.

The dimensions come from the supplied GIF and remain estimates. The motion
reference must already traverse this world-space hurdle; this file supplies
geometry and RSS contact dynamics, not a path planner.
"""

from __future__ import annotations

import robosimstudio as rss


def build_sim(asset, motion):
    start = motion.body_pos_w[0, 0]
    g1 = rss.EmbodimentSpec(
        name="g1_hurdle_repair",
        asset=rss.AssetSpec(path=str(asset)),
        preserve_free_bodies=True,
        enable_self_collisions=False,
        arms=(
            rss.ArmSpec(
                name="whole_body",
                joint_names=motion.joint_names,
                ee_body_name="right_wrist_yaw_link",
                home_q=tuple(float(value) for value in motion.joint_pos[0]),
            ),
        ),
    )
    scene = rss.Scene(cell="void", name="g1_hurdle_repair", physics=rss.MUJOCO_500HZ)
    scene.add(rss.robot(g1))
    scene.add(
        rss.slab(
            name="hurdle",
            half=(0.0325, 0.675, 0.04),
            at=(float(start[0] + 0.90), float(start[1]), 0.04),
            color=(0.48, 0.50, 0.51),
            friction=0.8,
        )
    )
    scene.add(
        rss.CameraSpec(
            name="review",
            width=960,
            height=540,
            eye=(float(start[0] + 1.55), float(start[1] - 2.45), 1.48),
            look_at=(float(start[0] + 0.45), float(start[1]), 0.65),
        )
    )
    return scene.build(control_mode="joint_position")
