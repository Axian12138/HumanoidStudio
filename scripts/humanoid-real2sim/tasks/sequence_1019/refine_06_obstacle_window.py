"""Optimize a protected pre-contact swing window against table and basket geometry."""

import argparse
import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np

from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields, refresh_fk, sha256


def main():
    import mujoco
    import trimesh
    from scipy.interpolate import CubicSpline, RegularGridInterpolator
    from scipy.optimize import least_squares

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed-motion", type=Path, help="Optional motion supplying spline initialization only.")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, args.output / "solver_snapshot.py")
    table_path = args.run / "review/scene/dining_table_world_v4.obj"
    cache = args.run / "review/scene/dining_table_corner_sdf.npz"
    axes = [np.arange(0.15, 0.651, 0.01), np.arange(-2.20, -1.849, 0.01), np.arange(-0.02, 0.771, 0.01)]
    if not cache.exists():
        table = trimesh.load_mesh(table_path, force="mesh")
        if not table.is_watertight:
            raise ValueError("Signed-distance optimization requires the closed table mesh")
        grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
        flat = grid.reshape(-1, 3)
        values = np.empty(len(flat))
        for start in range(0, len(flat), 2000):
            values[start : start + 2000] = trimesh.proximity.signed_distance(table, flat[start : start + 2000])
            if start % 20000 == 0:
                print("table grid", start, len(flat), flush=True)
        np.savez_compressed(cache, values=values.reshape(grid.shape[:-1]), table_sha256=sha256(table_path))
    with np.load(cache) as cached:
        if str(cached["table_sha256"]) != sha256(table_path):
            raise ValueError("SDF cache does not match table mesh")
        sdf = RegularGridInterpolator(tuple(axes), cached["values"], bounds_error=False, fill_value=-0.1)
    motion = MotionFields.load(args.candidate / "motion.npz", args.candidate / "motion_order.json")
    model_path = args.run / "scene/03_04_05/robot_props_drawer_with_rack_early_registered.xml"
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    names = [
        "left_" + name + "_joint" for name in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
    ]
    cols = [motion.joint_names.index(name) for name in names]
    addresses = np.array([int(model.joint(name).qposadr[0]) for name in motion.joint_names])
    lower, upper = np.array([model.joint(name).range for name in names]).T
    time = 27 + np.arange(len(motion.joint_pos)) / 50
    rows = np.flatnonzero((time >= 60.14 - 1e-7) & (time <= 61.4 + 1e-7))
    sample = np.unique(np.r_[np.arange(0, len(rows), 2), len(rows) - 1])
    prior = motion.joint_pos[np.ix_(rows, cols)].astype(float)
    colliders = []
    for geom in range(model.ngeom):
        body_name = model.body(int(model.geom_bodyid[geom])).name
        if not body_name.startswith("left_") or any(
            part in body_name for part in ("hand", "shoulder", "elbow", "wrist")
        ):
            continue
        if not (model.geom_contype[geom] or model.geom_conaffinity[geom]):
            continue
        size = model.geom_size[geom]
        if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_CAPSULE:
            raise ValueError(f"Unsupported leg collider {model.geom(geom).name}")
        # A capsule is the union of spheres along its centerline. Surface
        # vertices alone can miss a thin obstacle between adjacent vertices.
        points = np.zeros((31, 3))
        points[:, 2] = np.linspace(-size[1], size[1], len(points))
        colliders.append((geom, points, float(size[0])))
    leg_bodies = {int(model.geom_bodyid[geom]) for geom, _, _ in colliders}
    basket_body = model.body("round_laundry_basket").id
    basket_address = int(model.joint("round_laundry_basket_free").qposadr[0])
    with np.load(args.candidate / "round_laundry_basket.npz") as track:
        basket = np.c_[track["position_w"], track["quaternion_xyzw"][:, [3, 0, 1, 2]]]
    ankle = model.body("left_ankle_roll_link").id
    target_ankle = motion.body_pos_w[rows, motion.body_names.index("left_ankle_roll_link")]
    knots = np.linspace(time[rows[0]], time[rows[-1]], 16)
    evaluations = 0

    def values(x):
        delta = np.vstack([np.zeros(6), x.reshape(len(knots) - 2, 6), np.zeros(6)])
        return prior + CubicSpline(knots, delta, bc_type="clamped")(time[rows])

    def residual(x):
        nonlocal evaluations
        q = values(x)
        terms, worst = [], 0.0
        for ri in sample:
            k = rows[ri]
            data.qpos[:3] = motion.body_pos_w[k, 0]
            data.qpos[3:7] = motion.body_quat_w[k, 0]
            data.qpos[addresses] = motion.joint_pos[k]
            data.qpos[addresses[cols]] = q[ri]
            data.qpos[basket_address : basket_address + 7] = basket[k]
            mujoco.mj_forward(model, data)
            for geom, points, radius in colliders:
                world = points @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
                depths = np.maximum(0.0, sdf(world) + radius + 0.003)
                worst = max(worst, float(depths.max()))
                terms.extend(depths * 1500)
                terms.append(max(0.0, radius - float(world[:, 2].min())) * 1500)
            basket_depth = 0.0
            for contact in data.contact:
                first, second = map(int, model.geom_bodyid[[contact.geom1, contact.geom2]])
                other = second if first == basket_body else first
                if basket_body in (first, second) and other in leg_bodies:
                    basket_depth = max(basket_depth, -float(contact.dist) + 0.001)
            terms.append(basket_depth * 10000)
            terms.extend((data.xpos[ankle] - target_ankle[ri]) * 3)
        terms.extend(((q - prior) * 0.15).ravel())
        terms.extend((np.diff(q - prior, n=2, axis=0) * 30).ravel())
        terms.extend((np.maximum(lower - q, 0) * 10000).ravel())
        terms.extend((np.maximum(q - upper, 0) * 10000).ravel())
        terms.extend((np.maximum(abs(np.diff(q, axis=0)) * 50 - np.array([32, 32, 32, 20, 37, 37]), 0) * 10).ravel())
        evaluations += 1
        if evaluations % 100 == 0:
            print("eval", evaluations, "table_margin_depth", worst, flush=True)
        return np.asarray(terms)

    initial = np.zeros((len(knots) - 2) * 6)
    if args.seed_motion:
        with np.load(args.seed_motion) as seed_file:
            seed = seed_file["joint_pos"]
        if seed.shape != motion.joint_pos.shape or not np.isfinite(seed).all():
            raise ValueError("Seed must have matching dimensions and finite joint coordinates")
        seed_delta = seed[np.ix_(rows, cols)] - prior
        initial = np.stack([np.interp(knots[1:-1], time[rows], seed_delta[:, j]) for j in range(6)], axis=1).ravel()
    fit = least_squares(residual, initial, max_nfev=100, diff_step=1e-4, ftol=1e-7)
    joint_pos = motion.joint_pos.astype(float).copy()
    joint_pos[np.ix_(rows, cols)] = values(fit.x)
    robot_path = Path(
        "/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof_dex3/g1_29dof_dex3_offsethand.xml"
    )
    result = refresh_fk(replace(motion, joint_pos=joint_pos), robot_path)
    for name in (
        "motion_order.json",
        "source_sync.npz",
        "human_motion_world.npz",
        "body_prior.npz",
        "black_ribbon_hat.npz",
        "floor_strawberry.npz",
        "round_laundry_basket.npz",
        "drawer_slide.npz",
        "tracks_manifest.json",
    ):
        shutil.copy2(args.candidate / name, args.output / name)
    result.save(args.output / "motion.npz")
    report = {
        "status": "sampled_obstacle_optimization_pending_exact_validation",
        "source_candidate": str(args.candidate),
        "input_motion_sha256": sha256(args.candidate / "motion.npz"),
        "table_sha256": sha256(table_path),
        "grid_sha256": sha256(cache),
        "solver_snapshot_sha256": sha256(args.output / "solver_snapshot.py"),
        "seed_motion_sha256": sha256(args.seed_motion) if args.seed_motion else None,
        "model_sha256": sha256(model_path),
        "spline_knots_s": knots.tolist(),
        "solver_cost": float(fit.cost),
        "solver_optimality": float(fit.optimality),
        "solver_message": str(fit.message),
        "window_s": [float(time[rows[0]]), float(time[rows[-1]])],
        "solver_success": bool(fit.success),
        "nfev": fit.nfev,
        "maximum_joint_change_rad": float(abs(values(fit.x) - prior).max()),
        "limitations": [
            "10mm interpolated SDF and sampled capsule centerline spheres are optimization aids only.",
            "Exact100Hz room, basket contacts, joint contract, source similarity and support checks remain required.",
        ],
    }
    (args.output / "review.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report, flush=True)


if __name__ == "__main__":
    main()
