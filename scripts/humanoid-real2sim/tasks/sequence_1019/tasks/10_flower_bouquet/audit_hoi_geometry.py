"""Dense per-frame hand/forearm samples against declared convex scene proxies."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    import mujoco
    import trimesh
    from scipy.spatial import ConvexHull
    from scipy.spatial.transform import Rotation

    from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields

    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--motion", type=Path, required=True)
    p.add_argument("--tracks", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--assets", type=Path)
    p.add_argument("--penetration-only", action="store_true")
    p.add_argument("--side", choices=["left", "right"])
    p.add_argument("--source-range", type=float, nargs=2)
    a = p.parse_args()
    order = a.run.parents[1] / "curated_snapshot/retarget/ground_support_27_end/sidecars/motion_order.json"
    motion = MotionFields.load(a.motion, order)
    m = mujoco.MjModel.from_xml_path(
        "/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof_dex3/g1_29dof_dex3_offsethand.xml"
    )
    d = mujoco.MjData(m)
    addr = np.array([m.joint(x).qposadr[0] for x in motion.joint_names])
    folder = a.assets or a.run / "scene/10_flower_bouquet/registered_assets"
    objects = {}
    object_names = ["flower_bouquet", "small_wicker_basket", "cabinet_support", "cabinet_side"]
    if (folder / "room_back_wall.obj").exists():
        object_names.append("room_back_wall")
    for name in object_names:
        paths = (
            list(folder.glob(name + "_hull*.obj"))
            if name in ["flower_bouquet", "small_wicker_basket"]
            else [folder / (name + ".obj")]
        )
        vertices = [trimesh.load(x, force="mesh", process=False).vertices for x in paths]
        planes = [ConvexHull(v).equations for v in vertices]
        bounds = [(v.min(axis=0), v.max(axis=0)) for v in vertices]
        track = np.load(a.tracks / (name + ".npz")) if name in ["flower_bouquet", "small_wicker_basket"] else None
        objects[name] = (planes, track, bounds)
    samples = {}
    for g in range(m.ngeom):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if a.side and not name.startswith(a.side + "_"):
            continue
        if not name.endswith("_collision") or not any(
            name.startswith(s)
            for s in ["left_hand_", "right_hand_", "left_wrist", "right_wrist", "left_elbow_yaw", "right_elbow_yaw"]
        ):
            continue
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mi = m.geom_dataid[g]
            first = m.mesh_vertadr[mi]
            v = m.mesh_vert[first : first + m.mesh_vertnum[mi]].copy()
            samples[g] = (name, v)
        elif m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
            size = m.geom_size[g]
            samples[g] = (
                name,
                np.array([[x, y, z] for x in [-1, 0, 1] for y in [-1, 0, 1] for z in [-1, 0, 1]]) * size,
            )
        elif m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE:
            rad, half = m.geom_size[g, :2]
            v = []
            for z in np.linspace(-half, half, 9):
                for theta in np.linspace(0, 2 * np.pi, 24, endpoint=False):
                    v.append([rad * np.cos(theta), rad * np.sin(theta), z])
            for sign in [-1, 1]:
                for latitude in np.linspace(0, np.pi / 2, 7):
                    for theta in np.linspace(0, 2 * np.pi, 24, endpoint=False):
                        v.append(
                            [
                                rad * np.cos(latitude) * np.cos(theta),
                                rad * np.cos(latitude) * np.sin(theta),
                                sign * (half + rad * np.sin(latitude)),
                            ]
                        )
            samples[g] = (name, np.array(v))
    worst = {}
    series = []
    for k in range(len(motion.joint_pos)):
        if a.source_range and not (a.source_range[0] - 1e-8 <= 108 + k / 50 <= a.source_range[1] + 1e-8):
            continue
        d.qpos[:3] = motion.body_pos_w[k, 0]
        d.qpos[3:7] = motion.body_quat_w[k, 0]
        d.qpos[addr] = motion.joint_pos[k]
        mujoco.mj_forward(m, d)
        frame = {}
        for name, (planes, track, bounds) in objects.items():
            R = np.eye(3) if track is None else Rotation.from_quat(track["quaternion_xyzw"][k]).as_matrix()
            pos = np.zeros(3) if track is None else track["position_w"][k]
            minimum = 1.0
            which = ""
            for g, (label, v) in samples.items():
                w = v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
                local = (w - pos) @ R
                if a.penetration_only:
                    per_hull = []
                    for e, (low, high) in zip(planes, bounds, strict=True):
                        close = np.all((local >= low - 0.01) & (local <= high + 0.01), axis=1)
                        signed = np.full(len(local), 0.01)
                        signed[close] = np.max(local[close] @ e[:, :3].T + e[:, 3], axis=1)
                        per_hull.append(signed)
                    distance = np.min(per_hull, axis=0)
                else:
                    distance = np.min([np.max(local @ e[:, :3].T + e[:, 3], axis=1) for e in planes], axis=0)
                value = float(distance.min())
                if value < minimum:
                    minimum = value
                    which = label
                key = label + "::" + name
                if key not in worst or value < worst[key]["min_signed_halfspace_distance_m"]:
                    worst[key] = {"min_signed_halfspace_distance_m": value, "source_s": 108 + k / 50}
            frame[name] = {"min_signed_halfspace_distance_m": minimum, "geom": which}
        series.append(frame)
    a.out.write_text(
        json.dumps(
            {
                "status": "dense sampled geometry diagnostic, not exact continuous collision proof",
                "frames": len(series),
                "side": a.side,
                "source_range_s": a.source_range,
                "broadphase": "all points tested against each convex AABB expanded10mm; outside points certified nonpenetrating and reported as positive10mm lower bound"
                if a.penetration_only
                else "disabled",
                "sample_counts": {name: len(v) for name, v in samples.values()},
                "method": "all collision mesh vertices plus dense capsule/box boundary; union of convex halfspace signs; negative values certify sampled penetration, positive values are not exact Euclidean clearance",
                "worst_pairs": worst,
                "frames_data": series,
            },
            indent=2,
        )
    )
    print({name: min(x[name]["min_signed_halfspace_distance_m"] for x in series) for name in objects})


if __name__ == "__main__":
    main()
