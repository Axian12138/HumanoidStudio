"""Preserve the new crown grasp wrist target while avoiding enabled robot self contacts."""

import argparse
import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np

from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields, refresh_fk

MODEL = Path(
    "/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof_dex3/g1_29dof_dex3_offsethand.xml"
)


def main():
    import mujoco
    from scipy.interpolate import CubicSpline
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--candidate", default="03_04_05_joint_v58")
    p.add_argument("--output", default="03_04_05_joint_v60")
    p.add_argument("--exclude-evidenced-elbow-wrist", action="store_true")
    p.add_argument("--native-hulls", action="store_true")
    p.add_argument("--wrist-weight", type=float, default=1000)
    p.add_argument("--seed-candidate")
    a = p.parse_args()
    r = a.run
    c = r / "retarget/candidates" / a.candidate
    out = r / "retarget/candidates" / a.output
    out.mkdir(exist_ok=False)
    motion = MotionFields.load(c / "motion.npz", c / "motion_order.json")
    baseline = MotionFields.load(
        r / "retarget/candidates/03_04_05_joint_v50/motion.npz",
        r / "retarget/candidates/03_04_05_joint_v50/motion_order.json",
    )
    model_path = r / "scene/03_04_05/robot_native_shoulder_hulls_diagnostic.xml" if a.native_hulls else MODEL
    m = mujoco.MjModel.from_xml_path(str(model_path))
    d = mujoco.MjData(m)
    addr = np.array([m.joint(n).qposadr[0] for n in motion.joint_names])
    names = [
        "right_" + n + "_joint"
        for n in ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"]
    ]
    cols = [motion.joint_names.index(n) for n in names]
    lo = np.array([m.joint(n).range[0] for n in names])
    hi = np.array([m.joint(n).range[1] for n in names])
    w = m.body("right_wrist_yaw_link").id
    wi = motion.body_names.index("right_wrist_yaw_link")
    q = motion.joint_pos.copy().astype(float)
    geoms = [
        g
        for g in range(m.ngeom)
        if m.geom_contype[g]
        and any(m.body(m.geom_bodyid[g]).name.startswith("right_" + x) for x in ["shoulder", "elbow", "wrist"])
    ]
    pairs = sorted(
        {
            tuple(sorted((g, j)))
            for g in geoms
            for j in range(m.ngeom)
            if m.geom_contype[j] and m.geom_bodyid[g] != m.geom_bodyid[j]
        }
    )
    pm = {v: i for i, v in enumerate(pairs)}

    native_pair = (
        tuple(sorted((m.geom("native_hull_torso_link").id, m.geom("native_hull_right_shoulder_yaw_link").id)))
        if a.native_hulls
        else None
    )

    def contact_depth():
        dep = np.zeros(len(pairs))
        for con in d.contact:
            pair = tuple(sorted((int(con.geom1), int(con.geom2))))
            if a.exclude_evidenced_elbow_wrist and {m.geom(pair[0]).name, m.geom(pair[1]).name} == {
                "right_elbow_yaw_collision",
                "right_wrist_collision",
            }:
                continue
            if pair in pm:
                dep[pm[pair]] = max(dep[pm[pair]], -float(con.dist))
        if native_pair is not None:
            dep[pm[native_pair]] = max(0, -mujoco.mj_geomDistance(m, d, *native_pair, 0.1, None) + 0.001)
        return dep

    times = 27 + np.arange(len(q)) / 50
    rows = np.flatnonzero((times >= 39.1) & (times <= 41.5))
    sample = np.unique(np.r_[np.arange(0, len(rows), 3), len(rows) - 1])
    knots = np.linspace(39.1, 41.5, 9)
    limits = []
    for ri in sample:
        k = rows[ri]
        d.qpos[:3] = motion.body_pos_w[k, 0]
        d.qpos[3:7] = motion.body_quat_w[k, 0]
        d.qpos[addr] = baseline.joint_pos[k]
        mujoco.mj_forward(m, d)
        limit = contact_depth() + 0.0005
        if native_pair is not None:
            limit[pm[native_pair]] = 0
        limits.append(limit)
    prior = q[np.ix_(rows, cols)]

    def values(x):
        return prior + CubicSpline(knots, np.vstack([np.zeros(7), x.reshape(7, 7), np.zeros(7)]), bc_type="clamped")(
            times[rows]
        )

    evaluations = 0

    def measure(v):
        positions = []
        rotations = []
        depths = []
        for si, ri in enumerate(sample):
            k = rows[ri]
            d.qpos[:3] = motion.body_pos_w[k, 0]
            d.qpos[3:7] = motion.body_quat_w[k, 0]
            d.qpos[addr] = q[k]
            d.qpos[addr[cols]] = v[ri]
            mujoco.mj_forward(m, d)
            targetr = Rotation.from_quat(motion.body_quat_w[k, wi][[1, 2, 3, 0]])
            positions.append(d.xpos[w] - motion.body_pos_w[k, wi])
            rotations.append((Rotation.from_matrix(d.xmat[w].reshape(3, 3)) * targetr.inv()).as_rotvec())
            depths.append(np.maximum(contact_depth() - limits[si], 0))
        return np.array(positions), np.array(rotations), np.array(depths)

    def residual(x):
        nonlocal evaluations
        v = values(x)
        pos, rot, dep = measure(v)
        evaluations += 1
        if evaluations % 100 == 0:
            print(
                {
                    "evaluations": evaluations,
                    "max_self_excess": float(dep.max()),
                    "max_wrist_error": float(np.linalg.norm(pos, axis=1).max()),
                },
                flush=True,
            )
        return np.r_[
            (pos * a.wrist_weight).ravel(),
            (rot * (a.wrist_weight * 0.15)).ravel(),
            (dep * 4000).ravel(),
            ((v - prior) * 0.15).ravel(),
            (np.diff(v - prior, n=2, axis=0) * 50).ravel(),
            (np.maximum(abs(np.diff(v, axis=0)) * 50 - np.array([37] * 5 + [22] * 2), 0) * 30).ravel(),
            (np.maximum(lo - v, 0) * 1000).ravel(),
            (np.maximum(v - hi, 0) * 1000).ravel(),
        ]

    initial = np.zeros(49)
    if a.seed_candidate:
        seed = MotionFields.load(
            r / "retarget/candidates" / a.seed_candidate / "motion.npz",
            r / "retarget/candidates" / a.seed_candidate / "motion_order.json",
        )
        initial = CubicSpline(times[rows], seed.joint_pos[np.ix_(rows, cols)] - prior)(knots[1:-1]).ravel()
    fit = least_squares(residual, initial, max_nfev=70, ftol=1e-6, diff_step=1e-4)
    v = values(fit.x)
    pos, rot, dep = measure(v)
    q[np.ix_(rows, cols)] = v
    records = {
        "solver_success": bool(fit.success),
        "excluded_proxy_pair": ["right_elbow_yaw_collision", "right_wrist_collision"]
        if a.exclude_evidenced_elbow_wrist
        else None,
        "proxy_evidence": "review/stage3/04_native_right_elbow_wrist/report.json",
        "global_model_contact_filters_changed": False,
        "native_mesh_hull_constraint": a.native_hulls,
        "max_wrist_error_m": float(np.linalg.norm(pos, axis=1).max()),
        "max_rotation_error_rad": float(np.linalg.norm(rot, axis=1).max()),
        "max_self_excess_m": float(dep.max()),
        "max_speed_rad_s": float(abs(np.diff(v, axis=0)).max() * 50),
        "sample_stride": 3,
        "nfev": fit.nfev,
    }
    result = refresh_fk(replace(motion, joint_pos=q), MODEL)
    for f in c.iterdir():
        if (
            f.is_file()
            and f.suffix in [".npz", ".json", ".xml"]
            and f.name not in ["motion.npz", "review.json", "manifest.json"]
            and not f.name.startswith(("collision_", "grasp_"))
        ):
            shutil.copy2(f, out / f.name)
    result.save(out / "motion.npz")
    (out / "review.json").write_text(
        json.dumps(
            {
                "status": "self_clearance_candidate_pending_all_audits",
                "base": str(c),
                "window": [39.1, 41.5],
                "records": records,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
