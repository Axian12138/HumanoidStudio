"""Clip 10 scene with independently movable basket and bouquet.

The five basket solids leave the opening clear. Proxies are explicit estimates;
source-camera registration and independent settling must be audited separately.
"""

import json
import os
from pathlib import Path

import robosimstudio as rss

RUN = Path(os.environ["RSS_SEQUENCE_RUN"])
ASSETS = RUN / "scene/10_flower_bouquet/assets"
ROOT = Path(__file__).resolve().parents[6]
ROOM = ROOT / "outputs/humanoid-real2sim/2026-10-20/sequence_1019/curated_snapshot/assets/ikea_1012_room.obj"
FLOWER_POSITION = (-2.572504044, 0.58258009, 0.712115347)
FLOWER_QUAT = (0.6412218182, -0.0448385959, -0.0534365622, 0.7641783915)
BASKET_POSITION = (-2.640000105, -0.319999993, 0.662)
BASKET_QUAT = (0.0, 0.0, 0.7071067812, 0.7071067812)


def make_scene(robot=None, room=True):
    scene = rss.Scene(cell="void", name="flower_basket_1019", physics=rss.MUJOCO_500HZ)
    if robot is not None:
        scene.add(rss.robot(robot))
    if room:
        scene.add(
            rss.StaticSpec(name="room_visual", asset=rss.AssetSpec(path=str(ROOM)), world_xform=True, collide=False)
        )
    basket = json.loads((ASSETS.parent / "basket_multiframe_candidate.json").read_text())
    # Candidate dimensions and support follow multiview/image evidence.

    scene.add(
        rss.StaticSpec(
            name="cabinet_support",
            asset=rss.AssetSpec(path=str(ASSETS / "cabinet_support_v2.obj")),
            world_xform=True,
            collide=True,
        )
    )
    scene.add(
        rss.StaticSpec(
            name="cabinet_side",
            asset=rss.AssetSpec(path=str(ASSETS / "cabinet_side_v2.obj")),
            world_xform=True,
            collide=True,
        )
    )
    for name, pos, quat, mass in [
        ("flower_bouquet", FLOWER_POSITION, FLOWER_QUAT, 0.15),
        ("small_wicker_basket", tuple(basket["position_m"]), tuple(basket["quaternion_xyzw"]), 0.35),
    ]:
        mesh_name = name + "_estimate" if name == "small_wicker_basket" else name
        scene.add(
            rss.body(
                name=name,
                asset=rss.AssetSpec(
                    path=str(ASSETS / f"{mesh_name}.obj"),
                    collision_paths=tuple(str(p) for p in sorted(ASSETS.glob(mesh_name + "_hull*.obj"))),
                ),
                at=pos,
                quat=quat,
                mass=mass,
                friction=0.6,
            )
        )
    # Source right camera from independent measured rigid, not retarget wrist.

    import numpy as np
    from scipy.spatial.transform import Rotation

    track = np.load(RUN / "inputs/10_flower_bouquet/rigid/wrist_reference.npz")
    cal = json.loads(
        Path(
            "/home/galbot/workspace/ScaleTrack/data/1019test/lined-rect-basket-kinematic-v1/metadata/camera_calibration.json"
        ).read_text()
    )
    k = np.array(cal["cameras"]["right_wrist"]["K"])
    head = json.loads((RUN / "scene/10_flower_bouquet/head_camera_rigid_position_candidate.json").read_text())
    h = np.array(head["camera_world_transform"])
    kh = np.array(cal["cameras"]["head"]["K"])
    scene.add(
        rss.CameraSpec(
            name="head_estimate_110",
            width=1280,
            height=800,
            fx=float(kh[0, 0]),
            fy=float(kh[1, 1]),
            cx=float(kh[0, 2]),
            cy=float(kh[1, 2]),
            eye=tuple(h[:3, 3]),
            look_at=tuple(h[:3, 3] + h[:3, 2]),
            up=tuple(-h[:3, 1]),
        )
    )
    multi = json.loads((RUN / "scene/10_flower_bouquet/head_multiframe_candidates.json").read_text())[0]
    for ix, seconds in enumerate([110, 111]):
        h = np.array(multi["camera_world_transforms"][ix])
        scene.add(
            rss.CameraSpec(
                name=f"head_multi_{seconds}",
                width=1280,
                height=800,
                fx=float(kh[0, 0]),
                fy=float(kh[1, 1]),
                cx=float(kh[0, 2]),
                cy=float(kh[1, 2]),
                eye=tuple(h[:3, 3]),
                look_at=tuple(h[:3, 3] + h[:3, 2]),
                up=tuple(-h[:3, 1]),
            )
        )
    for source_s in [112.0, 113.0]:
        i = round((source_s - 108) * 50)
        r = Rotation.from_quat(track["right_rigid_quat_xyzw"][i]).as_matrix()
        t = np.eye(4)
        t[:3, :3] = r
        t[:3, 3] = track["right_rigid_pos_w"][i]
        t = t @ track["T_right_rigid_real_camera_optical"]
        scene.add(
            rss.CameraSpec(
                name=f"source_right_{int(source_s)}",
                width=1280,
                height=800,
                fx=float(k[0, 0]),
                fy=float(k[1, 1]),
                cx=float(k[0, 2]),
                cy=float(k[1, 2]),
                eye=tuple(t[:3, 3]),
                look_at=tuple(t[:3, 3] + t[:3, 2]),
                up=tuple(-t[:3, 1]),
            )
        )
        t = np.eye(4)
        t[:3, :3] = r
        t[:3, 3] = track["right_rigid_pos_w"][i]
        t = t @ np.linalg.inv(track["T_right_rigid_real_camera_optical"])
        scene.add(
            rss.CameraSpec(
                name=f"inverse_right_{int(source_s)}",
                width=1280,
                height=800,
                fx=float(k[0, 0]),
                fy=float(k[1, 1]),
                cx=float(k[0, 2]),
                cy=float(k[1, 2]),
                eye=tuple(t[:3, 3]),
                look_at=tuple(t[:3, 3] + t[:3, 2]),
                up=tuple(-t[:3, 1]),
            )
        )
    scene.add(rss.CameraSpec(name="review", width=960, height=540, eye=(-0.9, -1.8, 1.9), look_at=(-2.6, 0.0, 0.85)))
    return scene


def build_sim(asset, motion):
    robot = rss.EmbodimentSpec(
        name="g1_flower_1019",
        asset=rss.AssetSpec(path=str(asset)),
        preserve_free_bodies=True,
        enable_self_collisions=False,
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
