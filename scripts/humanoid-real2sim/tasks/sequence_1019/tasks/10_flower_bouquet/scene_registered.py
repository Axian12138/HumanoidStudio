"""Private clip 10 estimated scene; calibrated mount, level support and separate props."""

import json
import os
from pathlib import Path

import robosimstudio as rss

RUN = Path(os.environ["RSS_SEQUENCE_RUN"])
FOLDER = RUN / "scene/10_flower_bouquet"
ASSETS = Path(os.environ.get("RSS_FLOWER_ASSETS", str(FOLDER / "registered_assets")))


def make_scene(robot=None, room=True):
    import numpy as np
    from scipy.spatial.transform import Rotation

    scene = rss.Scene(cell="void", name="flower_basket_registered_candidate", physics=rss.MUJOCO_500HZ)
    if robot is not None:
        scene.add(rss.robot(robot))
    static_assets = [("room", False), ("cabinet_support", True), ("cabinet_side", True)]
    if (ASSETS / "room_back_wall.obj").exists():
        static_assets.append(("room_back_wall", True))
    for name, collision in static_assets:
        if name == "room" and not room:
            continue
        asset_path = ASSETS / (name + ".obj")
        if name == "room":
            unified_room = RUN / "scene/unified/per_furniture_v4.obj"
            asset_path = Path(os.environ.get("RSS_FLOWER_ROOM_MESH", str(unified_room)))
            if not asset_path.exists():
                asset_path = ASSETS / "room.obj"
        scene.add(
            rss.StaticSpec(
                name=name + "_visual" if name == "room" else name,
                asset=rss.AssetSpec(path=str(asset_path)),
                world_xform=True,
                collide=collision,
            )
        )
    properties_path = ASSETS / "physical_properties.json"
    properties = json.loads(properties_path.read_text()) if properties_path.exists() else {}
    for name, candidate, mass in [
        ("small_wicker_basket", "basket_registered_candidate.json", 0.35),
        ("flower_bouquet", "flower_registered_candidate.json", 0.15),
    ]:
        data = json.loads((FOLDER / candidate).read_text())
        base = ASSETS
        collision_base = ASSETS
        if os.environ.get("RSS_FLOWER_INITIAL_TRACKS"):
            track = np.load(Path(os.environ["RSS_FLOWER_INITIAL_TRACKS"]) / (name + ".npz"))
            frame = int(os.environ.get("RSS_FLOWER_INITIAL_FRAME", "-1"))
            data["position_m"] = track["position_w"][frame].tolist()
            data["quaternion_xyzw"] = track["quaternion_xyzw"][frame].tolist()
        scene.add(
            rss.body(
                name=name,
                asset=rss.AssetSpec(
                    path=str(base / (name + ".obj")),
                    material_colors=True,
                    collision_paths=tuple(str(p) for p in sorted(collision_base.glob(name + "_hull*.obj"))),
                ),
                at=tuple(data["position_m"]),
                quat=tuple(data["quaternion_xyzw"]),
                mass=properties.get("mass_kg", {}).get(name, mass),
                friction=0.6,
            )
        )
    mount = json.loads((FOLDER / "head_marker_diagnostic_v4_header.json").read_text())
    E = np.array(mount["T_head_rigid_camera_optical"])
    K = np.array(mount["K"])
    raw = json.loads((RUN / "inputs/10_flower_bouquet/head_raw_pose_audit_header.json").read_text())["rows"]
    C = Rotation.from_euler("z", 90, degrees=True).as_matrix() @ np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    for seconds in [109.4, 110.0, 111.0, 112.0, 113.0, 114.0, 120.0, 123.4, 124.6, 125.0, 128.0]:
        row = raw[str(seconds)]
        H = np.eye(4)
        H[:3, :3] = C @ Rotation.from_quat(row["quaternion_xyzw_raw"]).as_matrix()
        H[:3, 3] = C @ row["position_raw"]
        H[2, 3] += 0.08245383203029633
        T = H @ E
        scene.add(
            rss.CameraSpec(
                name=f"head_registered_{seconds:g}",
                width=1280,
                height=800,
                fx=float(K[0, 0]),
                fy=float(K[1, 1]),
                cx=float(K[0, 2]),
                cy=float(K[1, 2]),
                eye=tuple(T[:3, 3]),
                look_at=tuple(T[:3, 3] + T[:3, 2]),
                up=tuple(-T[:3, 1]),
            )
        )
    scene.add(rss.CameraSpec(name="review", width=960, height=540, eye=(-0.9, -1.8, 1.9), look_at=(-2.6, 0.1, 0.85)))
    return scene


def build_sim(asset, motion):
    robot = rss.EmbodimentSpec(
        name="g1_flower_1019",
        asset=rss.AssetSpec(path=str(asset)),
        preserve_free_bodies=True,
        enable_self_collisions=True,
        arms=(
            rss.ArmSpec(
                name="whole_body",
                joint_names=motion.joint_names,
                ee_body_name="right_wrist_yaw_link",
                home_q=tuple(float(x) for x in motion.joint_pos[0]),
            ),
        ),
    )
    return make_scene(robot).build(control_mode="joint_position")


scene = make_scene() if os.environ.get("RSS_FLOWER_LIVE") == "1" else None
