"""Fit a two-hand, palm-down kinematic carry of the lined rectangular basket.

The object path is estimated from the independent rigid markers. HOI geometry,
not marker wrist targets, determines hand placement. This does not run physics.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation
import trimesh

from robosimstudio.humanoid_real2sim.paths import example_dir
from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields, refresh_fk, sha256


TASK = example_dir(__file__)
SNAPSHOT = TASK.parents[1]
ORDER = SNAPSHOT / "retarget/ground_support_27_end/sidecars/motion_order.json"
MODEL = Path(
    "/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/robots/"
    "g1_29dof_dex3/g1_29dof_dex3_offsethand.xml"
)
SOURCE_MESHES = Path("/home/galbot/workspace/ScaleTrack/data/1012test/hoi_stage2/scene/assets")
FPS = 50
START = 98.0
CONTACT_BEGIN = 98.8
CONTACT_END = 106.4
RELEASE_END = 107.2
HALF_LENGTH = 0.19
HALF_WIDTH = 0.13
HEIGHT = 0.14
TANGENTIAL_HOI_OFFSET_M = 0.05
# The operator permits about 30 mm of normal overlap during the loaded carry.
# Measure it from the outer face: an SDF alone tops out at half the 8 mm wall.
ARM_PRELOAD_TARGET_M = 0.027
PALM_PRELOAD_TARGET_M = 0.017
CONTACT_CLEARANCE_FLOOR_M = -0.030
CONTACT_OUTER_LIMIT_M = HALF_LENGTH - 0.027
SOLIDS = (
    ((0, 0, .004), (.19, .13, .004)),
    ((.186, 0, .074), (.004, .13, .066)),
    ((-.186, 0, .074), (.004, .13, .066)),
    ((0, .126, .074), (.182, .004, .066)),
    ((0, -.126, .074), (.182, .004, .066)),
)


def basket_sdf(points: np.ndarray) -> np.ndarray:
    """Signed distance to the union of basket base and 8 mm walls."""
    distances = []
    for midpoint, half in SOLIDS:
        delta = np.abs(points - midpoint) - half
        distances.append(np.linalg.norm(np.maximum(delta, 0), axis=1)
                         + np.minimum(np.max(delta, axis=1), 0))
    return np.min(distances, axis=0)


def collision_points(model: mujoco.MjModel, geom: int) -> np.ndarray:
    """Sample the actual hand, wrist and forearm collision primitives."""
    kind = model.geom_type[geom]
    if kind == mujoco.mjtGeom.mjGEOM_MESH:
        mesh = model.geom_dataid[geom]
        first = model.mesh_vertadr[mesh]
        return model.mesh_vert[first:first + model.mesh_vertnum[mesh]:450].copy()
    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        x, y, z = model.geom_size[geom]
        return np.asarray([[a, b, c] for a in (-x, 0, x)
                           for b in (-y, 0, y) for c in (-z, 0, z)])
    if kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
        radius, half = model.geom_size[geom, :2]
        angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        circles = []
        for axis in np.linspace(-half, half, 7):
            circles.extend((radius * np.cos(a), radius * np.sin(a), axis) for a in angles)
        for end in (-1, 1):
            for latitude in (np.pi / 6, np.pi / 3, np.pi / 2):
                radial = radius * np.cos(latitude)
                axis = end * (half + radius * np.sin(latitude))
                circles.extend((radial * np.cos(a), radial * np.sin(a), axis) for a in angles)
        return np.asarray(circles)
    raise ValueError(f"unsupported collision geom: {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom)}")


def frame(seconds: float) -> int:
    return round((seconds - START) * FPS)


def smoothstep(value: np.ndarray) -> np.ndarray:
    x = np.clip(value, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def estimate_basket(rigid: np.lib.npyio.NpzFile) -> tuple[np.ndarray, np.ndarray]:
    """Use bilateral side contact to estimate a level basket trajectory."""
    left = savgol_filter(rigid["left_rigid_pos_w"].astype(float), 21, 3, axis=0)
    right = savgol_filter(rigid["right_rigid_pos_w"].astype(float), 21, 3, axis=0)
    center = (left + right) / 2
    span = left[:, :2] - right[:, :2]
    yaw = np.unwrap(np.arctan2(span[:, 1], span[:, 0]))
    yaw = savgol_filter(yaw, 21, 3)
    base = center[:, 2] - 0.18  # calibrated at the 99 s dining-table contact
    # The object rests on the dining table before the bilateral carry and on
    # the coffee table after release. Both heights are measured scene priors.
    contact = frame(99.0)
    release = frame(106.7)
    center[:contact, :2] = center[contact, :2]
    center[release:, :2] = center[release, :2]
    yaw[:contact] = yaw[contact]
    yaw[release:] = yaw[release]
    base[:contact] = 0.754
    base[contact:release] = np.maximum(base[contact:release], 0.400)
    base[release:] = 0.400
    # A short vertical transition avoids copying tracker jitter into the
    # release/landing boundary.
    edge = frame(106.3)
    base[edge:release] = np.linspace(base[edge], 0.400, release - edge)
    center[:, 2] = base
    # The rigid markers sit on the dorsal side of the human wrists. Their
    # midpoint is only a pose prior, not a measured basket center. Slide the
    # basket tangentially so the reachable wrist and proximal forearm can
    # share the opposing side-wall contact bands seen in the source video.
    center[:, :2] += TANGENTIAL_HOI_OFFSET_M * np.column_stack((-np.sin(yaw), np.cos(yaw)))
    quat = Rotation.from_euler("z", yaw[:, None]).as_quat()
    return center.astype(np.float32), quat.astype(np.float32)


def export_visual_mesh(path: Path) -> None:
    pieces = [
        trimesh.load_mesh(SOURCE_MESHES / f"object_lined_rect_basket_{part}_0.obj")
        for part in range(1, 6)
    ]
    mesh = trimesh.util.concatenate(pieces)
    mesh.export(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rigid", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    prior = MotionFields.load(TASK / "body_prior.npz", ORDER)
    if len(prior.joint_pos) != 550 or prior.fps != FPS:
        raise ValueError("task 09 must span 98–109 s at 50 Hz")
    with np.load(args.rigid, allow_pickle=False) as rigid:
        if not np.all(rigid["left_valid"]) or not np.all(rigid["right_valid"]):
            raise ValueError("wrist rigid observations are not valid for every target frame")
        object_pos, object_quat = estimate_basket(rigid)
        camera_right_pos = rigid["right_camera_aligned_wrist_pos_w"].astype(float)
        camera_right_quat = rigid["right_camera_aligned_wrist_quat_xyzw"].astype(float)

    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    addresses = np.asarray([model.joint(name).qposadr[0] for name in prior.joint_names])
    joints = prior.joint_pos.astype(float).copy()
    keyframes = np.r_[
        [frame(t) for t in (98.0, 98.1, 98.2, 98.3, 98.4, 98.5, 98.6, 98.7)],
        np.arange(frame(CONTACT_BEGIN), frame(CONTACT_END) + 1, 5),
    ]
    if keyframes[-1] != frame(CONTACT_END):
        keyframes = np.r_[keyframes, frame(CONTACT_END)]
    fit_report = {}
    for side, sign in (("left", 1), ("right", -1)):
        arm_names = [f"{side}_{part}_joint" for part in (
            "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
            "wrist_roll", "wrist_pitch", "wrist_yaw",
        )]
        hand_names = [name for name in prior.joint_names if name.startswith(f"{side}_hand_")]
        names = arm_names + hand_names
        cols = np.asarray([prior.joint_names.index(name) for name in names])
        addr = addresses[cols]
        lower = np.asarray([model.joint(name).range[0] for name in names]) + 1e-5
        upper = np.asarray([model.joint(name).range[1] for name in names]) - 1e-5
        palm_id = model.geom(f"{side}_hand_palm_collision").id
        wrist_geom = model.geom(f"{side}_wrist_collision").id
        forearm_geom = model.geom(f"{side}_elbow_yaw_collision").id
        tip_ids = [model.body(f"{side}_hand_{part}_finger_tip").id for part in ("thumb", "index", "middle")]
        wrist_id = model.body(f"{side}_wrist_yaw_link").id
        face_sign = -1 if side == "left" else 1
        collision_geoms = []
        for geom in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom)
            if not name or not name.endswith("_collision") or not (
                name.startswith(f"{side}_hand_") or geom in (wrist_geom, forearm_geom)
            ):
                continue
            collision_geoms.append((geom, collision_points(model, geom)))
        offsets = []
        key_residuals = []
        previous = None
        for k in keyframes:
            data.qpos[:3] = prior.body_pos_w[k, 0]
            data.qpos[3:7] = prior.body_quat_w[k, 0]
            data.qpos[addresses] = prior.joint_pos[k]
            baseline = prior.joint_pos[k, cols].astype(float)
            basket_R = Rotation.from_quat(object_quat[k]).as_matrix()
            center = object_pos[k]
            time_s = START + k / FPS
            # The final approach tracks each x-side wall's outward normal in
            # reverse: an external hover moves inward without lateral sweep.
            approach_offset = .10 * np.clip((CONTACT_BEGIN - time_s) / .4, 0, 1)
            preload = float(smoothstep((time_s - CONTACT_BEGIN) / .4))
            # Both hands approach opposing outside walls. During the loaded
            # carry the wrist, forearm and palm may overlap the wall by up to
            # about 30 mm along its normal, as requested for Stage 4 grip.
            palm_local = np.array([sign * (0.226 - PALM_PRELOAD_TARGET_M * preload), -0.100, 0.105])
            wrist_local = np.array([sign * (0.226 - ARM_PRELOAD_TARGET_M * preload), 0.000, 0.090])
            forearm_local = np.array([sign * (0.228 - ARM_PRELOAD_TARGET_M * preload), 0.125, 0.090])
            tip_local = np.array([[sign * .197, -.100, .08],
                                  [sign * .212, -.070, .08],
                                  [sign * .212, -.125, .08]])
            for local in (palm_local, wrist_local, forearm_local):
                local[0] += sign * approach_offset
            tip_local[:, 0] += sign * approach_offset
            palm_target = center + basket_R @ palm_local
            wrist_target = center + basket_R @ wrist_local
            forearm_target = center + basket_R @ forearm_local
            tips_target = center + tip_local @ basket_R.T
            along_wall = basket_R @ np.array([0, -1, 0])
            # The body prior has closed fists. Give the solver room to open
            # digits, but keep each arm near the reviewed support motion.
            spread = np.array([0.90, 0.90, 0.90, 1.00, 0.55, 0.55, 0.55] + [2.0] * len(hand_names))
            lo = np.maximum(lower, baseline - spread)
            hi = np.minimum(upper, baseline + spread)
            seed = np.clip(baseline if previous is None else previous, lo + 1e-6, hi - 1e-6)

            def residual(q: np.ndarray) -> np.ndarray:
                data.qpos[addr] = q
                mujoco.mj_forward(model, data)
                palm = data.geom_xpos[palm_id]
                palm_mat = data.geom_xmat[palm_id].reshape(3, 3)
                face = face_sign * palm_mat[:, 0]
                tips = data.xpos[tip_ids]
                wrist_center = data.geom_xpos[wrist_geom]
                forearm_center = data.geom_xpos[forearm_geom]
                wrist_axis = data.geom_xmat[wrist_geom].reshape(3, 3)[:, 2]
                # Right camera orientation is a weak 6D prior because the
                # supplied cam_mat USD differs from the offsethand model's
                # camera variant. Contact geometry governs the final pose.
                camera = np.zeros(6)
                if side == "right":
                    camera[:3] = 0.6 * (data.xpos[wrist_id] - camera_right_pos[k])
                    camera[3:] = 0.08 * (
                        Rotation.from_quat(camera_right_quat[k]).inv()
                        * Rotation.from_matrix(data.xmat[wrist_id].reshape(3, 3))
                    ).as_rotvec()
                clearance = []
                sidewall = []
                surface_dists = []
                for geom, sampled in collision_geoms:
                    world = sampled @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
                    local = (world - center) @ basket_R
                    distances = basket_sdf(local)
                    clearance_floor = .0015 * (1 - preload) + CONTACT_CLEARANCE_FLOOR_M * preload
                    clearance.append(np.maximum(clearance_floor - distances, 0))
                    surface_dists.append(np.min(distances))
                    # A thin wall may pass through an oriented box without
                    # containing vertices. Limit normal overlap directly
                    # from the outer face, including after crossing the wall.
                    outer_limit = .1915 * (1 - preload) + CONTACT_OUTER_LIMIT_M * preload
                    sidewall.append(max(outer_limit - np.min(sign * local[:, 0]), 0)
                                    if np.min(local[:, 2]) < .145 else 0.0)
                closest = np.min(surface_dists)
                # Bring the grasp target in smoothly. A hard switch at 98.4 s made the wrist
                # snap despite collision-free sampled keyframes.
                target_weight = float(smoothstep((time_s - START) / .6))
                return np.r_[
                    target_weight * np.array([55, 100, 100]) * (basket_R.T @ (palm - palm_target)),
                    target_weight * np.array([75, 120, 120]) * (basket_R.T @ (wrist_center - wrist_target)),
                    target_weight * np.array([30, 60, 60]) * (basket_R.T @ (forearm_center - forearm_target)),
                    target_weight * 7 * (face - np.array([0, 0, -1])),
                    target_weight * 8 * (palm_mat[:, 2] - along_wall),
                    target_weight * 6 * (wrist_axis - basket_R @ np.array([0, 1, 0])),
                    target_weight * 10 * (tips - tips_target).ravel(),
                    0.6 * (q[:4] - baseline[:4]),
                    1.0 * (q[4:7] - baseline[4:7]),
                    0.12 * (q[7:] - baseline[7:]),
                    0.35 * (q - previous) if previous is not None else np.zeros(len(q)),
                    camera,
                    600 * np.concatenate(clearance),
                    2000 * np.asarray(sidewall),
                    target_weight * 40 * (closest - .002),
                ]

            solution = least_squares(residual, seed, bounds=(lo, hi), max_nfev=75)
            previous = solution.x.copy()
            offsets.append(solution.x - baseline)
            data.qpos[addr] = solution.x
            mujoco.mj_forward(model, data)
            palm = data.geom_xpos[palm_id]
            face = face_sign * data.geom_xmat[palm_id].reshape(3, 3)[:, 0]
            tips = data.xpos[tip_ids]
            key_residuals.append({
                "time_s": START + k / FPS,
                "palm_target_error_m": float(np.linalg.norm(palm - palm_target)),
                "wrist_target_error_m": float(np.linalg.norm(data.geom_xpos[wrist_geom] - wrist_target)),
                "forearm_target_error_m": float(np.linalg.norm(data.geom_xpos[forearm_geom] - forearm_target)),
                "palm_down_error_deg": float(np.rad2deg(np.arccos(np.clip(np.dot(face, [0, 0, -1]), -1, 1)))),
                "tip_target_error_m": np.linalg.norm(tips - tips_target, axis=1).tolist(),
            })
        # Fade only the interaction joints in and out. Root, legs, torso and
        # opposite support contacts remain exactly the reviewed body prior.
        knot_frames = np.r_[keyframes, frame(RELEASE_END), len(joints) - 1]
        knot_offsets = np.vstack((offsets, np.zeros(len(cols)), np.zeros(len(cols))))
        correction = PchipInterpolator(knot_frames, knot_offsets, axis=0)(np.arange(len(joints)))
        joints[:, cols] = np.clip(joints[:, cols] + correction, lower, upper)
        fit_report[side] = {
            "max_arm_change_rad": float(np.max(np.abs(correction[:, :7]))),
            "max_hand_change_rad": float(np.max(np.abs(correction[:, 7:]))),
            "keys": sorted(key_residuals, key=lambda item: item["time_s"]),
        }

    candidate = replace(prior, joint_pos=joints.astype(np.float32),
                        joint_vel=np.gradient(joints, 1 / FPS, axis=0).astype(np.float32))
    candidate = refresh_fk(candidate, MODEL)
    candidate.save(args.out / "motion.npz")
    np.savez_compressed(
        args.out / "object_estimate.npz", position_w=object_pos,
        quaternion_xyzw=object_quat, fps=np.asarray(FPS),
        source_range_s=np.asarray([98.0, 109.0]),
        trajectory_source=np.asarray("bilateral_rigid_contact_kinematic_estimate_no_physics"),
    )
    export_visual_mesh(args.out / "basket_visual.obj")
    with np.load(TASK / "human.npz", allow_pickle=False) as human:
        positions = np.asarray(human["keypoints_w"], dtype=np.float32)
    names = (
        "pelvis", "left_hip", "right_hip", "left_knee", "right_knee",
        "left_ankle", "right_ankle", "left_sole", "right_sole",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hand_aux", "right_hand_aux",
        "pelvis_aux", "left_elbow_twist_aux", "right_elbow_twist_aux",
    )
    edges = np.asarray([(names.index(a), names.index(b)) for a, b in (
        ("pelvis", "left_hip"), ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
        ("left_ankle", "left_sole"), ("pelvis", "right_hip"), ("right_hip", "right_knee"),
        ("right_knee", "right_ankle"), ("right_ankle", "right_sole"),
        ("pelvis", "left_shoulder"), ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"), ("left_wrist", "left_hand_aux"),
        ("pelvis", "right_shoulder"), ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"), ("right_wrist", "right_hand_aux"),
        ("left_shoulder", "right_shoulder"),
    )], dtype=np.int32)
    np.savez_compressed(
        args.out / "human_motion_world.npz", positions_w=positions,
        joint_names=np.asarray(names), edges=edges, fps=np.asarray(FPS),
        source_range_s=np.asarray([98.0, 109.0]),
    )
    report = {
        "stage": "3.2_kinematic_candidate_unreviewed",
        "source_window_s": [98.0, 109.0],
        "contact_window_s": [CONTACT_BEGIN, CONTACT_END],
        "release_complete_s": RELEASE_END,
        "basket_dimensions_m": [0.38, 0.26, HEIGHT],
        "basket_tangential_hoi_offset_m": TANGENTIAL_HOI_OFFSET_M,
        "arm_preload_target_m": ARM_PRELOAD_TARGET_M,
        "palm_preload_target_m": PALM_PRELOAD_TARGET_M,
        "contact_clearance_floor_m": CONTACT_CLEARANCE_FLOOR_M,
        "contact_outer_limit_m": CONTACT_OUTER_LIMIT_M,
        "basket_source_initial_pose_m": [-0.775000095, -2.569999933, 0.754000008],
        "basket_contact_fitted_initial_pose_m": object_pos[0].tolist(),
        "basket_final_pose_m": object_pos[-1].tolist(),
        "fitting": fit_report,
        "input_sha256": {str(path): sha256(path) for path in (
            TASK / "body_prior.npz", TASK / "human.npz", TASK / "source_sync.npz",
            args.rigid, MODEL,
        )},
        "limitations": [
            "Basket pose follows bilateral rigid wrists; not measured object 6D or physical motion.",
            "No tracker, contact force, friction, release stability or training-data approval.",
            "Right camera-mapped wrist is weakly weighted; left real rigid-to-camera extrinsic unavailable.",
        ],
    }
    (args.out / "fit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(args.out), "initial": object_pos[0].tolist(),
                      "final": object_pos[-1].tolist(),
                      "palm_error_max_m": {s: max(k["palm_target_error_m"] for k in v["keys"])
                                           for s, v in fit_report.items()}}, indent=2))


if __name__ == "__main__":
    main()
