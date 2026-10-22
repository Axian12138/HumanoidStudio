"""Measure actual distal/palm mesh proximity and opposed contacts at every source frame."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    import mujoco
    from scipy.spatial.transform import Rotation, Slerp

    from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields

    parser = argparse.ArgumentParser()
    for name in ["run", "motion", "tracks", "assets", "out"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--sample-hz", type=float)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    order = args.run.parents[1] / "curated_snapshot/retarget/ground_support_27_end/sidecars/motion_order.json"
    motion = MotionFields.load(args.motion, order)
    model = mujoco.MjModel.from_xml_path(
        "/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof_dex3/g1_29dof_dex3_offsethand.xml"
    )
    state = mujoco.MjData(model)
    addresses = np.array([model.joint(name).qposadr[0] for name in motion.joint_names])
    flower = np.load(args.tracks / "flower_bouquet.npz")
    basket = np.load(args.tracks / "small_wicker_basket.npz")
    joint_positions = motion.joint_pos
    root_positions = motion.body_pos_w[:, 0]
    root_quaternions = motion.body_quat_w[:, 0]
    sample_hz = args.sample_hz or float(motion.fps)
    if sample_hz != float(motion.fps):
        source_times = flower["source_time_s"]
        times = source_times[0] + np.arange(round((source_times[-1] - source_times[0]) * sample_hz) + 1) / sample_hz
        times = np.clip(times, source_times[0], source_times[-1])

        def interpolate(values):
            return np.column_stack([np.interp(times, source_times, column) for column in values.T])

        joint_positions = interpolate(joint_positions)
        root_positions = interpolate(root_positions)
        root_quaternions = Slerp(source_times, Rotation.from_quat(root_quaternions[:, [1, 2, 3, 0]]))(times).as_quat()[
            :, [3, 0, 1, 2]
        ]

        def track_at_times(track):
            return {
                "source_time_s": times,
                "position_w": interpolate(track["position_w"]),
                "quaternion_xyzw": Slerp(source_times, Rotation.from_quat(track["quaternion_xyzw"]))(times).as_quat(),
            }

        flower, basket = track_at_times(flower), track_at_times(basket)
    profile = json.loads((args.assets / "wrapper_profile.json").read_text())
    input_paths = [
        args.motion,
        order,
        args.tracks / "flower_bouquet.npz",
        args.tracks / "small_wicker_basket.npz",
        args.assets / "wrapper_profile.json",
        args.run / "scene/10_flower_bouquet/basket_registered_candidate.json",
        Path(
            "/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/robots/"
            "g1_29dof_dex3/g1_29dof_dex3_offsethand.xml"
        ),
    ]
    provenance = {
        "inputs": [
            {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in input_paths
        ],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "parsed_model_geometry_arrays_sha256": {
            name: hashlib.sha256(np.asarray(getattr(model, name)).tobytes()).hexdigest()
            for name in ("mesh_vert", "mesh_face", "geom_pos", "geom_quat", "geom_size", "geom_dataid")
        },
        "sample_fps": sample_hz,
        "interpolation": "Linear joint/root/object positions; SLERP root/object rotations when sample rate differs.",
        "method_limit": "Authored collision mesh vertices, not complete visual surface contact area or a force closure proof.",
    }
    slope = (profile["upper_radius_m"] - profile["lower_radius_m"]) / profile["height_m"]
    dims = json.loads((args.run / "scene/10_flower_bouquet/basket_registered_candidate.json").read_text())[
        "dimensions_world_aligned_m"
    ]
    meshes = {}
    for side in ["right", "left"]:
        for part in ["thumb_2", "index_1", "middle_1", "palm"]:
            name = f"{side}_hand_{part}_collision"
            geom = model.geom(name).id
            mesh = model.geom_dataid[geom]
            if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_BOX:
                half = model.geom_size[geom]
                meshes[name] = (
                    geom,
                    np.array(
                        [
                            [x, y, z]
                            for x in [-half[0], 0, half[0]]
                            for y in [-half[1], 0, half[1]]
                            for z in [-half[2], 0, half[2]]
                        ]
                    ),
                )
                continue
            meshes[name] = (
                geom,
                model.mesh_vert[model.mesh_vertadr[mesh] : model.mesh_vertadr[mesh] + model.mesh_vertnum[mesh]].copy(),
            )

    def world_points(name):
        geom, vertices = meshes[name]
        return vertices @ state.geom_xmat[geom].reshape(3, 3).T + state.geom_xpos[geom]

    records = []
    for k, t in enumerate(flower["source_time_s"]):
        if not (113.2 <= t <= 123.2 or 118.7 <= t <= 120.4 or 125.0 <= t <= 127.8):
            continue
        state.qpos[:3] = root_positions[k]
        state.qpos[3:7] = root_quaternions[k]
        state.qpos[addresses] = joint_positions[k]
        mujoco.mj_forward(model, state)
        row = {"source_s": float(t)}
        if 113.2 <= t <= 123.2:
            rot = Rotation.from_quat(flower["quaternion_xyzw"][k]).as_matrix()
            pads = {}
            normals = []
            patches = []
            for part in ["thumb_2", "index_1", "middle_1"]:
                local = (world_points(f"right_hand_{part}_collision") - flower["position_w"][k]) @ rot
                selected = local[(local[:, 2] >= 0.175) & (local[:, 2] <= 0.325)]
                if not len(selected):
                    pads[part] = {"gap_m": None, "reason": "no pad mesh in grip-height band"}
                    normals.append(None)
                    patches.append(np.empty((0, 3)))
                    continue
                radii = np.linalg.norm(selected[:, :2], axis=1)
                signed = (radii - profile["lower_radius_m"] - slope * selected[:, 2]) / np.sqrt(1 + slope * slope)
                touching = selected[np.abs(signed) <= 0.003]
                patch_normals = np.c_[
                    touching[:, :2] / np.maximum(np.linalg.norm(touching[:, :2], axis=1)[:, None], 1e-9),
                    np.full(len(touching), -slope),
                ]
                patch_normals /= np.maximum(np.linalg.norm(patch_normals, axis=1)[:, None], 1e-9)
                _, patch_indices = np.unique(np.round(patch_normals, 2), axis=0, return_index=True)
                patches.append(patch_normals[patch_indices])
                index = np.argmin(abs(signed))
                point = selected[index]
                normal = np.r_[point[:2] / max(np.linalg.norm(point[:2]), 1e-9), -slope]
                normal /= np.linalg.norm(normal)
                normals.append(normal)
                pads[part] = {
                    "gap_m": float(abs(signed[index])),
                    "signed_gap_m": float(signed[index]),
                    "point_local": point.tolist(),
                }
            opposed = [
                float(normals[0] @ normals[j]) if normals[0] is not None and normals[j] is not None else None
                for j in [1, 2]
            ]
            patch_opposed = [
                float(np.min(patches[0] @ patches[j].T)) if len(patches[0]) and len(patches[j]) else None
                for j in [1, 2]
            ]
            row["right"] = {
                "pads": pads,
                "thumb_opposition_normal_dots": opposed,
                "contact_patch_opposition_min_dot": patch_opposed,
                "contact_patch_normal_counts": [len(x) for x in patches],
                "patch_method": "all pad vertices within3mm of frustum surface in grip band; direction bins rounded.01 choose original exact unit-normal representatives",
            }
        if 118.7 <= t <= 120.4 or 125.0 <= t <= 127.8:
            pull = t < 121
            yaw = Rotation.from_quat(basket["quaternion_xyzw"][k]).as_euler("xyz")[2] - np.pi / 2
            rot = Rotation.from_euler("z", yaw).as_matrix()
            pads = {}
            for part in ["thumb_2", "index_1", "middle_1", "palm"]:
                local = (world_points(f"left_hand_{part}_collision") - basket["position_w"][k]) @ rot
                selected = local
                plane = dims[0] / 2 - (0.012 if pull and part in ["index_1", "middle_1"] else 0)
                if len(selected):
                    inward_face = pull and part in ["index_1", "middle_1"]
                    outward_gap = (selected[:, 0] - plane) * (-1 if inward_face else 1)
                    on_face = (abs(selected[:, 1]) <= dims[1] / 2) & (selected[:, 2] >= 0) & (selected[:, 2] <= dims[2])
                    face_patch = on_face & (outward_gap >= -0.002) & (outward_gap <= 0.003)
                    face_distance = np.sqrt(
                        (selected[:, 0] - plane) ** 2
                        + np.maximum(abs(selected[:, 1]) - dims[1] / 2, 0) ** 2
                        + np.maximum(selected[:, 2] - dims[2], 0) ** 2
                        + np.maximum(-selected[:, 2], 0) ** 2
                    )
                    index = np.argmin(face_distance)
                    pads[part] = {
                        "gap_m": float(face_distance[index]),
                        "contact_region": "edge"
                        if selected[index, 2] > dims[2]
                        or selected[index, 2] < 0
                        or abs(selected[index, 1]) > dims[1] / 2
                        else "face",
                        "point_basket_rectangle": selected[index].tolist(),
                        "face_patch_vertex_count": int(np.sum(face_patch)),
                        "face_outward_normal_rectangle": [-1, 0, 0] if inward_face else [1, 0, 0],
                    }
                else:
                    pads[part] = {"gap_m": None, "reason": "no mesh point opposite the rim or wall"}
            row["left"] = {
                "phase": "pull" if pull else "push",
                "pads": pads,
                "thumb_index_opposed_face_patches": bool(
                    pull
                    and pads["thumb_2"].get("face_patch_vertex_count", 0)
                    and pads["index_1"].get("face_patch_vertex_count", 0)
                ),
                "patch_scope": "actual pad vertices within finite wall face and signed outward gap [-2,3]mm; independent triangle-core collision audit required",
            }
        records.append(row)
    right = [r["right"] for r in records if "right" in r]
    summaries = {}
    for part in ["thumb_2", "index_1", "middle_1"]:
        gaps = [r["pads"][part]["gap_m"] for r in right]
        finite = [g for g in gaps if g is not None]
        summaries[part] = {
            "frames": len(gaps),
            "missing": sum(g is None for g in gaps),
            "within_3mm": sum(g is not None and g <= 0.003 for g in gaps),
            "max_gap_m": max(finite) if finite else None,
        }
    args.out.write_text(
        json.dumps(
            {
                "status": "geometric contact evidence only; intersecting meshes must still fail independent collision audit",
                "provenance": provenance,
                "right_summary": summaries,
                "records": records,
            },
            indent=2,
        )
    )
    print(json.dumps(summaries))


if __name__ == "__main__":
    main()
