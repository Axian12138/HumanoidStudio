"""Independent table strawberry/basket scene; five separate basket colliders."""

import json
import math
import os
from pathlib import Path

import robosimstudio as rss

REPO = Path(__file__).resolve().parents[6]
RUN = Path(
    os.environ.get(
        "RSS_SEQUENCE_RUN", REPO / "outputs/humanoid-real2sim/2026-10-20/sequence_1019/runs/21-08-52.042738+0800"
    )
)
ASSETS = Path("/home/galbot/workspace/ScaleTrack/data/1012test/hoi_stage2/scene/assets")
ROOM = REPO / "outputs/humanoid-real2sim/2026-10-20/sequence_1019/curated_snapshot/assets/ikea_1012_room.obj"


def make_scene():
    import numpy as np

    geometry_path = Path(
        os.environ.get("RSS_08_GEOMETRY_CONFIG", RUN / "scene/08_table_strawberry/geometry_active.json")
    )
    geometry = json.loads(geometry_path.read_text()) if geometry_path.exists() else {}
    transform = np.array(geometry.get("T_room_correction", np.eye(4)))
    yaw = math.atan2(transform[1, 0], transform[0, 0])
    room_quat = (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))
    scene = rss.Scene(cell="void", name="table_strawberry_1019", physics=rss.MUJOCO_500HZ)
    scene.add(
        rss.StaticSpec(
            name="room",
            asset=rss.AssetSpec(path=str(geometry.get("room_world_mesh", ROOM))),
            xform=(0, 0, 0, 0, 0, 0, 1) if geometry.get("room_world_mesh") else (*transform[:3, 3], *room_quat),
            world_xform=True,
            collide=False,
        )
    )
    # Preserve the accepted 07 terminal chair state. Its open mesh is audited
    # independently; do not replace the seat/legs with one solid convex hull.
    scene.add(
        rss.StaticSpec(
            name="grey_armchair",
            asset=rss.AssetSpec(path=str(ROOM.parents[1] / "tasks/07_armchair/grey_armchair_visual.obj")),
            xform=(-0.5523157716, -2.1457662582, 0.001, 0.0, 0.0, -0.0494105928, 0.9987785220),
            world_xform=True,
            collide=False,
        )
    )
    berry_pose = {"position": [-0.822655201, -2.20, 0.7907203], "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0]}
    if geometry:
        berry_pose = {
            "position": geometry["berry_position_world_m"],
            "quaternion_xyzw": geometry["berry_quaternion_xyzw"],
        }
    if os.environ.get("RSS_TABLE_STRAWBERRY_INITIAL_POSE"):
        berry_pose = json.loads(Path(os.environ["RSS_TABLE_STRAWBERRY_INITIAL_POSE"]).read_text())
    scene.add(
        rss.StaticSpec(
            name="dining_table_support_proxy",
            half_extents=(0.8, 0.425472975, 0.02),
            xform=(*(transform[:3, :3] @ [-0.23, -2.544527025, 0.734] + transform[:3, 3]), *room_quat),
            world_xform=True,
            visible=False,
        )
    )
    scene.add(
        rss.body(
            name="table_strawberry",
            asset=rss.AssetSpec(
                path=str(ASSETS / "object_strawberry_existing_0_0.obj"), approximate="convex_hull", keep_visual=True
            ),
            at=tuple(berry_pose["position"]),
            quat=tuple(berry_pose["quaternion_xyzw"]),
            mass=0.06,
            friction=0.7,
            color=(0.8, 0.07, 0.12),
        )
    )
    scene.add(
        rss.body(
            name="lined_rect_basket",
            asset=rss.AssetSpec(
                path=str(RUN / "scene/08_table_strawberry/basket.obj"),
                collision_paths=tuple(str(ASSETS / f"object_lined_rect_basket_{i}_0.obj") for i in range(1, 6)),
            ),
            at=(-0.92494571, -2.60162902, 0.754),
            quat=(0.0, 0.0, 0.71868527, 0.69533551),
            mass=0.6,
            friction=0.7,
        )
    )
    scene.add(rss.CameraSpec(name="review", width=960, height=640, eye=(-0.4, -0.8, 2.0), look_at=(-0.9, -2.3, 0.78)))
    camera_evidence = RUN / "review/scene/08_marker_camera/berry_multiview.json"
    if geometry and camera_evidence.exists():
        calibration = json.loads((RUN / "scene/10_flower_bouquet/head_marker_diagnostic_v4_header.json").read_text())
        intrinsic = np.array(calibration["K"])
        for time, values in json.loads(camera_evidence.read_text())["T_world_camera"].items():
            camera = np.array(values)
            scene.add(
                rss.CameraSpec(
                    name="head_" + time.replace(".", "_"),
                    width=1280,
                    height=800,
                    fx=intrinsic[0, 0],
                    fy=intrinsic[1, 1],
                    cx=intrinsic[0, 2],
                    cy=intrinsic[1, 2],
                    eye=tuple(camera[:3, 3]),
                    look_at=tuple(camera[:3, 3] + camera[:3, 2]),
                    up=tuple(-camera[:3, 1]),
                )
            )
    return scene


def build_sim(asset, motion):
    scene = make_scene()
    robot = rss.EmbodimentSpec(
        name="g1_table_strawberry",
        asset=rss.AssetSpec(path=str(asset)),
        preserve_free_bodies=True,
        enable_self_collisions=False,
        arms=(
            rss.ArmSpec(
                name="whole_body",
                joint_names=motion.joint_names,
                ee_body_name="right_wrist_yaw_link",
                home_q=tuple(float(v) for v in motion.joint_pos[0]),
            ),
        ),
    )
    scene.add(rss.robot(robot))
    return scene.build(control_mode="joint_position")


scene = make_scene()
