"""Refit the basket path after an initial -X placement, keeping robot motion fixed."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROBOT_XML = Path(
    "/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof_dex3/g1_29dof_dex3_offsethand.xml"
)
ROBOT_FIELDS = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")
FPS = 50


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mesh_points(model: mujoco.MjModel, name: str) -> tuple[int, np.ndarray]:
    import mujoco
    from scipy.spatial.transform import Rotation

    body = model.body(name).id
    pieces = []
    for geom in range(model.ngeom):
        if (
            model.geom_bodyid[geom] != body
            or model.geom_group[geom] == 3
            or model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH
        ):
            continue
        mid = model.geom_dataid[geom]
        mesh = model.mesh_vert[model.mesh_vertadr[mid] : model.mesh_vertadr[mid] + model.mesh_vertnum[mid]]
        pieces.append(Rotation.from_quat(model.geom_quat[geom][[1, 2, 3, 0]]).apply(mesh) + model.geom_pos[geom])
    return body, np.concatenate(pieces)


def main() -> None:
    import mujoco
    from scipy.interpolate import PchipInterpolator
    from scipy.optimize import least_squares

    parser = argparse.ArgumentParser()
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--motion-order", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--initial-offset-x", type=float, default=-0.05)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    order = json.loads(args.motion_order.read_text())
    with np.load(args.motion, allow_pickle=False) as src:
        robot = {name: src[name] for name in ROBOT_FIELDS}
        original = src["object_pos_w"][:, 0].astype(np.float64)
        quaternion_wxyz = src["object_quat_w"][:, 0].astype(np.float64)
        fps = int(src["fps"])
    if fps != FPS or len(original) != 850:
        raise ValueError("expected 850 frames at 50 Hz")
    model = mujoco.MjModel.from_xml_path(str(ROBOT_XML))
    data = mujoco.MjData(model)
    joint_addresses = np.asarray([model.joint(name).qposadr[0] for name in order["robot_joint_names"]])
    foot, foot_mesh = mesh_points(model, "left_ankle_roll_link")
    shin, shin_mesh = mesh_points(model, "left_knee_link")

    def surfaces_at(frame: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        data.qpos[:3] = robot["body_pos_w"][frame, 0]
        data.qpos[3:7] = robot["body_quat_w"][frame, 0]
        data.qpos[joint_addresses] = robot["joint_pos"][frame]
        mujoco.mj_forward(model, data)
        surfaces = []
        for body, mesh in ((foot, foot_mesh), (shin, shin_mesh)):
            world = mesh @ data.xmat[body].reshape(3, 3).T + data.xpos[body]
            world = world[(world[:, 2] > 0.07) & (world[:, 2] < 0.35)][::5]
            if len(world) == 0:
                raise ValueError(f"empty contact surface at frame {frame}")
            surfaces.append(world)
        return surfaces[0], surfaces[1], data.xpos[foot, :2].copy()

    def gaps(center: np.ndarray, surfaces: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        result = []
        for surface in surfaces:
            radius = np.interp(surface[:, 2], [0, 0.1, 0.3, 0.4], [0.105, 0.14, 0.165, 0.195])
            result.append(np.min(np.linalg.norm(surface[:, :2] - center, axis=1) - radius))
        return np.asarray(result)

    initial = original[0, :2].copy()
    initial[0] += args.initial_offset_x
    contact_start = None
    for frame in range(round((60.7 - 59) * FPS), round((63.7 - 59) * FPS) + 1):
        foot_surface, shin_surface, _ = surfaces_at(frame)
        if gaps(initial, (foot_surface, shin_surface))[0] <= 0.003:
            contact_start = frame
            break
    if contact_start is None:
        raise ValueError("foot never reaches the shifted basket")

    knot_frames = np.arange(contact_start, round((63.7 - 59) * FPS) + 1, 5, dtype=int)
    if knot_frames[-1] != round((63.7 - 59) * FPS):
        knot_frames = np.r_[knot_frames, round((63.7 - 59) * FPS)]
    previous = initial.copy()
    previous_ankle = None
    fitted_centers = []
    contact_log = []
    for frame in knot_frames:
        foot_surface, shin_surface, ankle = surfaces_at(int(frame))
        surfaces = (foot_surface, shin_surface)
        expected = previous if previous_ankle is None else previous + 0.75 * (ankle - previous_ankle)
        lower = np.asarray([max(-0.2, previous[0] - 0.025), max(-2.3, previous[1] - 0.0005)])
        upper = np.asarray([min(initial[0] + 0.005, previous[0] + 0.0005), min(-1.75, previous[1] + 0.025)])

        def residual(center: np.ndarray, surfaces=surfaces, expected=expected) -> np.ndarray:
            foot_gap, shin_gap = gaps(center, surfaces)
            return np.r_[
                80 * foot_gap,
                20 * min(shin_gap, 0.02),
                120 * min(foot_gap + 0.002, 0),
                120 * min(shin_gap + 0.002, 0),
                5 * (center - expected),
            ]

        result = least_squares(
            residual, np.clip(expected, lower + 1e-6, upper - 1e-6), bounds=(lower, upper), max_nfev=120
        )
        center = result.x
        fitted_centers.append(center)
        contact_log.append(
            {
                "source_frame": int(frame),
                "recording_time_s": 59 + frame / FPS,
                "center_xy_m": center.tolist(),
                "foot_shin_gap_m": gaps(center, surfaces).tolist(),
            }
        )
        previous, previous_ankle = center, ankle
    fitted_centers = np.asarray(fitted_centers)
    endpoint = fitted_centers[-1]
    path = original.copy()
    hold_end = contact_start - 1
    return_start = round((69 - 59) * FPS)
    return_old_anchor = original[return_start, :2]
    keys = np.r_[0, hold_end, knot_frames, round((64.1 - 59) * FPS), return_start]
    centers = np.vstack((initial, initial, fitted_centers, endpoint, endpoint))
    if not np.all(np.diff(keys) > 0):
        raise ValueError("interpolation keys are not strictly increasing")
    before = np.arange(return_start)
    path[before, :2] = PchipInterpolator(keys, centers, axis=0)(before)
    path[return_start:, :2] += endpoint - return_old_anchor
    path[:, 2] = original[:, 2]
    if not np.allclose(path[0, :2], initial, atol=1e-9):
        raise AssertionError("initial basket placement changed during interpolation")

    np.savez_compressed(
        args.out / "object_estimate.npz",
        position_w=path.astype(np.float32),
        quaternion_xyzw=quaternion_wxyz[:, [1, 2, 3, 0]].astype(np.float32),
        fps=np.asarray(FPS),
        status=np.asarray("fixed_humanoid_contact_refit_minus_x_5cm"),
    )
    np.savez_compressed(
        args.out / "basket_kinematic_trajectory.npz",
        position_w=path.astype(np.float32),
        quaternion_wxyz=quaternion_wxyz.astype(np.float32),
        linear_velocity_w=np.gradient(path, 1 / FPS, axis=0).astype(np.float32),
        source_frame=np.arange(len(path)),
        fps=np.asarray(FPS),
    )
    review = {
        "status": "kinematic_basket_refit_pending_visual_and_physics_review",
        "method": "same foot/shin surface contact geometry and ankle-motion prior as original fit; robot pose held fixed; hold until foot reaches shifted basket; preserve old return shape translated to new outbound endpoint",
        "source_motion": str(args.motion),
        "source_motion_sha256": sha256(args.motion),
        "robot_fields_unchanged": list(ROBOT_FIELDS),
        "robot_motion_recomputed": False,
        "initial_offset_world_x_m": args.initial_offset_x,
        "old_initial_xy_m": original[0, :2].tolist(),
        "new_initial_xy_m": initial.tolist(),
        "foot_contact_start_source_frame": int(contact_start),
        "foot_contact_start_recording_time_s": 59 + contact_start / FPS,
        "outbound_endpoint_xy_m": endpoint.tolist(),
        "return_path_offset_xy_m": (endpoint - return_old_anchor).tolist(),
        "selected_positions_m": {str(k): path[k].tolist() for k in (0, 100, 150, 200, 235, 400, 500, 700, 849)},
        "contact_knots": contact_log,
        "limitations": [
            "contact-conditioned kinematic estimate, not measured object tracking",
            "right forearm and left hand robot motion was not refit to the new object path",
            "table is visual context without collision in Stage 4",
        ],
    }
    (args.out / "kinematic_review.json").write_text(json.dumps(review, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: review[k]
                for k in (
                    "new_initial_xy_m",
                    "foot_contact_start_source_frame",
                    "outbound_endpoint_xy_m",
                    "return_path_offset_xy_m",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
