"""Continue bounded left-leg poses through the protected pre-contact swing."""

import argparse
import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np

from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields, refresh_fk, sha256


def main():
    import mujoco
    from scipy.interpolate import RegularGridInterpolator
    from scipy.optimize import minimize

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed-motion", type=Path)
    parser.add_argument("--self-proxy-mode", choices=("preserve", "measure-only"), default="preserve")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, args.output / "solver_snapshot.py")
    motion = MotionFields.load(args.candidate / "motion.npz", args.candidate / "motion_order.json")
    model_path = args.run / "scene/03_04_05/robot_props_drawer_with_rack_early_registered.xml"
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    addresses = np.array([int(model.joint(n).qposadr[0]) for n in motion.joint_names])
    names = ["left_" + n + "_joint" for n in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]
    cols = [motion.joint_names.index(n) for n in names]
    lower, upper = np.array([model.joint(n).range for n in names]).T
    speed = np.array([32, 32, 32, 20, 37, 37]) * 0.95
    legs = []
    for g in range(model.ngeom):
        body = model.body(model.geom_bodyid[g]).name
        if not body.startswith("left_") or any(x in body for x in ("hand", "shoulder", "elbow", "wrist")):
            continue
        if not (model.geom_contype[g] or model.geom_conaffinity[g]):
            continue
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_CAPSULE:
            raise ValueError("Unexpected noncapsule leg geometry")
        points = np.zeros((41, 3))
        points[:, 2] = np.linspace(-model.geom_size[g, 1], model.geom_size[g, 1], len(points))
        legs.append((g, points, model.geom_size[g, 0]))
    basket_body = model.body("round_laundry_basket").id
    basket_geoms = [
        g
        for g in range(model.ngeom)
        if model.geom_bodyid[g] == basket_body and (model.geom_contype[g] or model.geom_conaffinity[g])
    ]
    basket_address = int(model.joint("round_laundry_basket_free").qposadr[0])
    with np.load(args.candidate / "round_laundry_basket.npz") as track:
        basket = np.c_[track["position_w"], track["quaternion_xyzw"][:, [3, 0, 1, 2]]]
    cache_path = args.run / "review/scene/dining_table_corner_sdf.npz"
    table_path = args.run / "review/scene/dining_table_world_v4.obj"
    with np.load(cache_path) as cache:
        if str(cache["table_sha256"]) != sha256(table_path):
            raise ValueError("Wrong table cache")
        grid = RegularGridInterpolator(
            [np.arange(0.15, 0.651, 0.01), np.arange(-2.20, -1.849, 0.01), np.arange(-0.02, 0.771, 0.01)],
            cache["values"],
            bounds_error=False,
            fill_value=-0.1,
        )
    times = 27 + np.arange(len(motion.joint_pos)) / 50
    rows = np.flatnonzero((times > 60.14 + 1e-7) & (times < 61.4 - 1e-7))
    end = round((61.4 - 27) * 50)
    q_all = motion.joint_pos.astype(float).copy()
    seed_q = None
    if args.seed_motion:
        with np.load(args.seed_motion) as saved:
            seed_q = saved["joint_pos"]
        if seed_q.shape != q_all.shape or not np.isfinite(seed_q).all():
            raise ValueError("Seed shape or values invalid")
    reports = []
    self_pair = (model.geom("left_hip_collision").id, model.geom("pelvis_collision").id)
    contexts = []

    def constraints(q):
        values = []
        for state, blend, self_lower in contexts:
            data.qpos[:] = state
            data.qpos[addresses[cols]] = blend * q + (1 - blend) * previous
            mujoco.mj_forward(model, data)
            for geom, points, radius in legs:
                world = points @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
                values.extend(-grid(world) - radius - 0.0015)
                values.append(float(world[:, 2].min()) - radius - 0.0002)
                distance = 0.2
                for bg in basket_geoms:
                    bound = np.linalg.norm(data.geom_xpos[geom] - data.geom_xpos[bg])
                    bound -= model.geom_rbound[geom] + model.geom_rbound[bg]
                    if bound <= 0.1:
                        distance = min(distance, mujoco.mj_geomDistance(model, data, geom, bg, 0.2, None))
                values.append(distance - 0.001)
            if args.self_proxy_mode == "preserve":
                values.append(mujoco.mj_geomDistance(model, data, *self_pair, 0.2, None) - self_lower)
        return np.asarray(values)

    for k in rows:
        data.qpos[:3] = motion.body_pos_w[k, 0]
        data.qpos[3:7] = motion.body_quat_w[k, 0]
        data.qpos[addresses] = motion.joint_pos[k]
        data.qpos[basket_address : basket_address + 7] = basket[k]
        q0 = motion.joint_pos[k, cols].astype(float)
        previous = q_all[k - 1, cols]
        previous_delta = previous - motion.joint_pos[k - 1, cols]
        current_state = data.qpos.copy()
        prior_state = current_state.copy()
        prior_state[:3] = motion.body_pos_w[k - 1, 0]
        prior_state[3:7] = motion.body_quat_w[k - 1, 0]
        prior_state[addresses] = motion.joint_pos[k - 1]
        prior_state[basket_address : basket_address + 7] = basket[k - 1]
        midpoint = 0.5 * (current_state + prior_state)
        midpoint[3:7] /= np.linalg.norm(midpoint[3:7])
        midpoint[basket_address + 3 : basket_address + 7] /= np.linalg.norm(
            midpoint[basket_address + 3 : basket_address + 7]
        )
        contexts = []
        for state, blend in ((current_state, 1.0), (midpoint, 0.5)):
            data.qpos[:] = state
            mujoco.mj_forward(model, data)
            self_lower = min(0.0, mujoco.mj_geomDistance(model, data, *self_pair, 0.2, None)) - 0.0001
            contexts.append((state, blend, self_lower))
        lo = np.maximum.reduce(
            [lower + 1e-7, previous - speed / 50, motion.joint_pos[end, cols] - speed * (end - k) / 50]
        )
        hi = np.minimum.reduce(
            [upper - 1e-7, previous + speed / 50, motion.joint_pos[end, cols] + speed * (end - k) / 50]
        )
        if np.any(lo >= hi):
            raise ValueError(f"Empty reachable bounds at {times[k]}")

        def objective(q, q0=q0, previous_delta=previous_delta):
            return float(np.sum((q - q0) ** 2) + 2 * np.sum((q - q0 - previous_delta) ** 2))

        bump = np.sin(np.pi * (times[k] - 60.14) / 1.26) ** 2
        seeds = [q0 + previous_delta, q0 + bump * np.array([-0.25, -0.12, 0, 0.4, -0.15, 0])]
        if seed_q is not None:
            seeds.insert(1, seed_q[k, cols])
        outcomes = []
        for seed in seeds:
            fit = minimize(
                objective,
                np.clip(seed, lo + 1e-10, hi - 1e-10),
                bounds=list(zip(lo, hi, strict=True)),
                constraints=[{"type": "ineq", "fun": constraints}],
                method="SLSQP",
                options={"maxiter": 120, "ftol": 1e-9, "eps": 1e-5},
            )
            violation = max(0.0, -float(constraints(fit.x).min()))
            outcomes.append((violation, objective(fit.x), fit))
            if violation < 1e-6:
                break
        violation, cost, fit = min(outcomes, key=lambda item: (item[0] > 1e-6, item[0] if item[0] > 1e-6 else item[1]))
        q_all[k, cols] = fit.x
        reports.append(
            {
                "source_s": float(times[k]),
                "constraint_violation_m": violation,
                "cost": cost,
                "solver_success": bool(fit.success),
                "solver_message": str(fit.message),
            }
        )
        print(reports[-1], flush=True)
        (args.output / "frame_checks.json").write_text(json.dumps(reports, indent=2) + "\n")
    robot = Path(
        "/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof_dex3/g1_29dof_dex3_offsethand.xml"
    )
    result = refresh_fk(replace(motion, joint_pos=q_all), robot)
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
        "status": "pending_independent_geometry_support_source_review",
        "source_candidate": str(args.candidate),
        "input_sha256": sha256(args.candidate / "motion.npz"),
        "solver_sha256": sha256(args.output / "solver_snapshot.py"),
        "table_sha256": sha256(table_path),
        "model_sha256": sha256(model_path),
        "self_proxy_mode": args.self_proxy_mode,
        "seed_motion_sha256": sha256(args.seed_motion) if args.seed_motion else None,
        "maximum_sampled_constraint_violation_m": max(x["constraint_violation_m"] for x in reports),
        "limitations": [
            "Sampled capsule centerlines and interpolated SDF are optimization aids.",
            "Requires exact100Hz room sweep, basket equality, joint contract, support and source review.",
        ],
    }
    (args.output / "review.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
