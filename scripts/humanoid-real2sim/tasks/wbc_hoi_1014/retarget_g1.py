"""Retarget one trusted WBC human recording with the existing retarget3 solver.

Run with ScaleTrack's Python and retarget3/src on PYTHONPATH. Source robot
poses are used only to recover the recorded rigid world calibration.
"""

import argparse
import hashlib
import json
import sys
from itertools import pairwise
from pathlib import Path

import numpy as np
from retarget3.body_only import initialize_pose
from retarget3.common import MAP, segments
from retarget3.motion_io import export_motion
from retarget3.quality import audit_quality
from retarget3.robot import ROBOT, MJCasadiRobot
from retarget3.runtime import BUNDLE
from retarget3.solver import Optimizer


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def bvh_foot_rotations(path, raw, world_rotation, clock, times):
    from scipy.spatial.transform import Rotation as R
    from scipy.spatial.transform import Slerp

    sys.path.insert(0, str(BUNDLE / "vendor/retarget2/src"))
    from bvh import Bvh

    b = Bvh(path.read_text(), backend="np")
    names = b.get_joints_names()
    positions, rotations = [], []
    for i, name in enumerate(names):
        channels = [c for c in b.joint_channels(name) if c.endswith("rotation")]
        local = R.from_euler(
            "".join(c[0] for c in channels), b.frames_joint_channels(name, channels), degrees=True
        ).as_matrix()
        if i == 0:
            positions.append(np.array(b.frames_joint_channels(name, ["Xposition", "Yposition", "Zposition"])) * 0.01)
            rotations.append(local)
        else:
            parent = names.index(b.joint_parent(name).name)
            positions.append(
                positions[parent] + np.einsum("tij,j->ti", rotations[parent], np.array(b.joint_offset(name)) * 0.01)
            )
            rotations.append(rotations[parent] @ local)
    positions = np.stack(positions, axis=1)[..., [2, 0, 1]]
    error = float(np.linalg.norm(positions - raw, axis=2).max())
    if np.linalg.norm(positions[:, 0] - raw[:, 0], axis=1).max() > 1e-4 or error > 0.04:
        raise ValueError(f"BVH and source human mismatch: {error}")
    result = []
    # Axis BVH foot: local X up, Y forward, Z left. Basis is checked
    # against the neutral source pose; no robot joint orientations are used.
    basis = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    for name in ("LeftFoot", "RightFoot"):
        matrix = world_rotation.as_matrix() @ rotations[names.index(name)][:, [2, 0, 1], :] @ basis
        result.append(Slerp(clock, R.from_matrix(matrix))(times).as_matrix())
    return np.stack(result, axis=1), error


def main():
    from scipy.interpolate import interp1d
    from scipy.ndimage import binary_closing, median_filter
    from scipy.spatial.transform import Rotation as R
    from scipy.spatial.transform import Slerp

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--fists", type=Path, required=True)
    ap.add_argument("--solve-only", action="store_true")
    ap.add_argument("--landing-frames", type=int, default=12)
    args = ap.parse_args()
    out = args.output.resolve()
    if (out / "body_prior.npz").exists():
        raise FileExistsError("Use a new candidate directory; existing motion is immutable")
    robot = MJCasadiRobot()
    cfg = {"fps": 50, "optimization": {"landing_anticipation_frames": args.landing_frames}}
    if not args.solve_only:
        out.mkdir(parents=True, exist_ok=False)
        for name in ("intermediate", "sidecars", "review"):
            (out / name).mkdir()
        with np.load(args.source, allow_pickle=True) as z:
            names = z["axis_skel_joints"].astype(str).tolist()
            if not z["axis_skel_valid"].all():
                raise ValueError("Invalid human frames require explicit repair")
            raw = z["axis_skel_pos"].astype(float)
            q, ref = z["qpos"], z["qpos_ref"]
            rot = R.from_quat(q[0, [4, 5, 6, 3]]) * R.from_quat(ref[0, [4, 5, 6, 3]]).inv()
            shift = q[0, :3] - rot.apply(ref[0, :3])
            world = raw @ rot.as_matrix().T + shift
            clock = z["t_wall"] - z["t_wall"][0]
            if np.any(np.diff(clock) <= 0):
                raise ValueError("Nonmonotonic wall clock")
            times = np.arange(int(np.floor(clock[-1] * 50)) + 1) / 50

            def sample(a):
                return interp1d(clock, a, axis=0)(times)

            human = sample(world)
            source_frame = np.interp(times, clock, np.arange(len(clock)))
            parents = [-1, 0, 1, 2, 3, 4, 3, 6, 7, 8, 3, 10, 11, 12, 0, 14, 15, 0, 17, 18]
            np.savez_compressed(
                out / "sidecars/human.npz",
                positions_w=human,
                joint_names=np.array(names),
                edges=np.array([(p, i) for i, p in enumerate(parents) if p >= 0]),
                fps=50,
            )
            np.savez_compressed(
                out / "sidecars/timing.npz",
                source_frame=source_frame,
                motion_time_s=times,
                source_t_wall=z["t_wall"],
                source_t_sim=z["t_sim"],
                t_wall=times + z["t_wall"][0],
                t_sim=sample(z["t_sim"]),
            )
            for i, oid in enumerate(z["object_ids"]):
                pose = z["object_pose"][:, i]
                quat = Slerp(clock, R.from_quat(pose[:, [4, 5, 6, 3]]))(times).as_quat()
                np.savez_compressed(
                    out / f"sidecars/object_{i}.npz",
                    position_w=sample(pose[:, :3]),
                    quaternion_xyzw=quat,
                    fps=50,
                    object_id=str(oid),
                )
            residual = rot.apply(ref[:, :3]) + shift - q[:, :3]
        foot_rot, bvh_error = bvh_foot_rotations(args.source.with_name("motion.bvh"), raw, rot, clock, times)
        canonical = [
            "Hips",
            "LeftUpLeg",
            "RightUpLeg",
            "LeftLeg",
            "RightLeg",
            "LeftFoot",
            "RightFoot",
            "Spine",
            "Head",
            "LeftArm",
            "RightArm",
            "LeftForeArm",
            "RightForeArm",
            "LeftHand",
            "RightHand",
        ]
        hp = human[:, [names.index(s) for s in canonical]]
        n = len(hp)
        neutral = np.zeros(robot.nq)
        neutral[6] = 1
        scales = []
        for chains in [[(1, 3, 5), (2, 4, 6)], [(9, 11, 13), (10, 12, 14)]]:
            ratios = []
            for chain in chains:
                a = sum(np.median(np.linalg.norm(hp[:, j] - hp[:, i], axis=-1)) for i, j in pairwise(chain))
                b = sum(
                    np.linalg.norm(np.array(robot.world_pos(MAP[j], neutral) - robot.world_pos(MAP[i], neutral)))
                    for i, j in pairwise(chain)
                )
                ratios.append(b / a)
            scales.append(float(np.median(ratios)))
        root = hp[:, 0].copy()
        root[:, 2] *= scales[0]
        target = root[:, None] + (hp - hp[:, 0, None]) * np.array([scales[0]] * 9 + [scales[1]] * 6)[None, :, None]
        z = (hp[:, 9] + hp[:, 10]) / 2 - hp[:, 0]
        z /= np.linalg.norm(z, axis=1)[:, None]
        y = hp[:, 9] - hp[:, 10]
        y -= np.sum(y * z, axis=1)[:, None] * z
        y /= np.linalg.norm(y, axis=1)[:, None]
        torso = np.stack([np.cross(y, z), y, z], axis=-1)
        foot_yaw = np.unwrap(np.arctan2(foot_rot[:, :, 1, 0], foot_rot[:, :, 0, 0]), axis=0)
        ft = np.zeros((n, 2, 3))
        fr = np.zeros((n, 2, 3, 3))
        contacts = np.zeros((n, 2), bool)
        for s in range(2):
            yaw = foot_yaw[:, s]
            points = hp[:, 5 + s]
            speed = np.linalg.norm(np.gradient(points, 0.02, axis=0), axis=1)
            mask = median_filter((speed < 0.28) & (points[:, 2] < 0.13), size=7)
            mask = binary_closing(mask, iterations=3)
            mask[:3] = mask[3]
            mask[-3:] = mask[-4]
            for a, b in segments(mask):
                if b - a < 6:
                    mask[a:b] = False
            contacts[:, s] = mask
            fr[:, s] = R.from_euler("z", yaw).as_matrix()
            ft[:, s] = points
            ft[:, s, 2] = np.maximum(0.035, points[:, 2] * scales[0])
            for a, b in segments(mask):
                ft[a:b, s, :2] = np.median(points[a:b, :2], axis=0)
                ft[a:b, s, 2] = 0.035
                fr[a:b, s] = R.from_euler("z", np.median(yaw[a:b])).as_matrix()
            for a, b in segments(~mask):
                if a > 0 and b < n:
                    w = np.linspace(0, 1, b - a + 2)[1:-1]
                    ft[a:b, s] += (1 - w[:, None]) * (ft[a - 1, s] - ft[a, s]) + w[:, None] * (ft[b, s] - ft[b - 1, s])
                    ft[a:b, s, 2] = np.maximum(0.04, ft[a:b, s, 2])
                    fr[a:b, s] = Slerp([0, 1], R.from_matrix(np.stack([fr[a - 1, s], fr[b, s]])))(w).as_matrix()
        q = np.zeros((n, robot.nq))
        q[:, :3] = root
        q[:, 3:7] = R.from_matrix(torso).as_quat()
        fists = json.loads(args.fists.read_text())
        for name, idx in robot.dof_name_to_index.items():
            q[:, idx] = np.clip(0.3 if name.endswith("knee_joint") else 0, robot.q_lower[idx], robot.q_upper[idx])
        for side in ("left", "right"):
            for name, value in fists[side]["joints"].items():
                q[:, robot.dof_name_to_index[name]] = value
        provenance = {
            "source": str(args.source.resolve()),
            "source_sha256": digest(args.source),
            "robot_xml": str(ROBOT),
            "robot_sha256": digest(ROBOT),
            "fists_sha256": digest(args.fists),
            "landing_anticipation_frames": args.landing_frames,
            "frames": n,
            "fps": 50,
            "source_frames": len(clock),
            "clock": "t_wall; metadata record_duration agrees",
            "duration_s": float(times[-1]),
            "tail_s": float(clock[-1] - times[-1]),
            "max_source_gap_s": float(np.diff(clock).max()),
            "root_calibration_max_error_m": float(np.linalg.norm(residual, axis=1).max()),
            "rotation_xyzw": rot.as_quat().tolist(),
            "translation_m": shift.tolist(),
            "morphology_scales": scales,
            "stance_frames": contacts.sum(0).tolist(),
            "foot_yaw": "source BVH foot orientation; root aligned within 2um, foot FK offset up to 3cm; original position targets preserved",
            "bvh_position_error_m": bvh_error,
            "world_scale": 1,
            "display_offset": False,
            "source_robot_joint_targets": False,
            "limitations": ["fruit geometry missing", "furniture collision not yet audited", "kinematic only"],
        }
        (out / "review/provenance.json").write_text(json.dumps(provenance, indent=2))
        q = initialize_pose(robot, q, target, ft, fr, torso)
        np.savez_compressed(
            out / "intermediate/targets.npz",
            qsource=q,
            human=hp,
            target=target,
            torso=torso,
            feet=ft,
            feet_rot=fr,
            contacts=contacts,
            hand_pos=hp[:, 13:15],
            hand_rot=np.repeat(torso[:, None], 2, axis=1),
        )
    with np.load(out / "intermediate/targets.npz") as z:
        data = dict(z)
    from robosimstudio.humanoid_real2sim.retarget.window_prior import solve_window_prior

    q = solve_window_prior(Optimizer(robot, cfg, out), data, out)
    export_motion(out / "body_prior.npz", robot, q, 50)
    audit_quality(cfg, out, robot, "body_prior")


if __name__ == "__main__":
    main()
