"""Avoid the coffee top while preserving stance ankle position and sole normal."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields, refresh_fk

MODEL = Path(
    "/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof_dex3/g1_29dof_dex3_offsethand.xml"
)


def main():
    import mujoco
    import trimesh
    from scipy.interpolate import CubicSpline
    from scipy.optimize import least_squares
    from scipy.spatial import ConvexHull
    from scipy.spatial.transform import Rotation

    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--candidate", default="03_04_05_joint_v46")
    p.add_argument("--version", default="v2")
    p.add_argument("--yaw-limit-deg", type=float, default=5)
    a = p.parse_args()
    r = a.run
    src = r / "retarget/candidates" / a.candidate
    out = r / "repairs/04_right_stance_coffee" / a.version
    out.mkdir(parents=True, exist_ok=False)
    motion = MotionFields.load(src / "motion.npz", src / "motion_order.json")
    mesh = trimesh.load_mesh(r / "scene/03_04_05/coffee_top_raw.obj")
    hull = ConvexHull(mesh.vertices)
    equations = np.unique(np.round(hull.equations, 10), axis=0)
    assert mesh.is_watertight
    assert abs(abs(mesh.volume) - hull.volume) < 1e-5
    m = mujoco.MjModel.from_xml_path(str(MODEL))
    d = mujoco.MjData(m)
    names = ["right_" + n + "_joint" for n in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]
    cols = [motion.joint_names.index(n) for n in names]
    addr = np.array([m.joint(n).qposadr[0] for n in motion.joint_names])
    lo, hi = np.array([m.joint(n).range for n in names]).T
    time = 27 + np.arange(len(motion.joint_pos)) / 50
    rows = np.flatnonzero((time >= 42.3) & (time <= 43.5))
    t = time[rows]
    sample = np.unique(np.r_[np.arange(0, len(rows), 2), len(rows) - 1])
    prior = motion.joint_pos[np.ix_(rows, cols)].astype(float)
    ankle = m.body("right_ankle_roll_link").id
    bi = motion.body_names.index("right_ankle_roll_link")
    targetP = motion.body_pos_w[rows, bi]
    targetR = Rotation.from_quat(motion.body_quat_w[rows, bi][:, [1, 2, 3, 0]])
    normals = targetR.apply(np.tile([0, 0, 1], (len(rows), 1)))
    colliders = []
    for g in range(m.ngeom):
        n = m.body(m.geom_bodyid[g]).name
        if not n.startswith("right_") or not any(x in n for x in ["hip", "knee", "ankle"]) or not m.geom_contype[g]:
            continue
        assert m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE
        pts = np.zeros((31, 3))
        pts[:, 2] = np.linspace(-m.geom_size[g, 1], m.geom_size[g, 1], 31)
        colliders.append((g, pts, m.geom_size[g, 0]))
    knots = np.linspace(42.3, 43.5, 10)
    calls = 0
    metrics = {}

    def values(x):
        return prior + CubicSpline(knots, np.vstack([np.zeros(6), x.reshape(8, 6), np.zeros(6)]), bc_type="clamped")(t)

    def residual(x):
        nonlocal calls, metrics
        q = values(x)
        terms = []
        margin = 0
        foot = 0
        normal = 0
        yaw = 0
        for ri in sample:
            k = rows[ri]
            d.qpos[:3] = motion.body_pos_w[k, 0]
            d.qpos[3:7] = motion.body_quat_w[k, 0]
            d.qpos[addr] = motion.joint_pos[k]
            d.qpos[addr[cols]] = q[ri]
            mujoco.mj_forward(m, d)
            for g, pts, rad in colliders:
                w = pts @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
                separation = np.max(w @ equations[:, :3].T + equations[:, 3], axis=1)
                depth = np.maximum(0, rad + 0.002 - separation)
                terms.extend(depth * 2000)
                margin = max(margin, float(depth.max()))
                terms.append(max(0, rad - w[:, 2].min()) * 1500)
            rot = Rotation.from_matrix(d.xmat[ankle].reshape(3, 3))
            pe = d.xpos[ankle] - targetP[ri]
            ne = rot.apply([0, 0, 1]) - normals[ri]
            re = (rot * targetR[ri].inv()).as_rotvec()
            terms.extend(pe * 15000)
            terms.extend(ne * 5000)
            terms.extend(re * 0.3)
            terms.append(max(0, abs(re[2]) - np.radians(a.yaw_limit_deg)) * 10000)
            foot = max(foot, float(np.linalg.norm(pe)))
            normal = max(normal, float(np.linalg.norm(ne)))
            yaw = max(yaw, float(abs(re[2])))
        terms.extend(((q - prior) * 0.3).ravel())
        terms.extend((np.diff(q - prior, n=2, axis=0) * 60).ravel())
        terms.extend((np.maximum(lo - q, 0) * 10000).ravel())
        terms.extend((np.maximum(q - hi, 0) * 10000).ravel())
        calls += 1
        metrics = {
            "margin_violation_m": margin,
            "ankle_position_error_m": foot,
            "sole_normal_error": normal,
            "foot_yaw_change_rad": yaw,
        }
        if calls % 100 == 0:
            print({"evaluations": calls, **metrics}, flush=True)
        return np.array(terms)

    fit = least_squares(residual, np.zeros(48), max_nfev=100, ftol=1e-7, diff_step=1e-4)
    q = values(fit.x)
    residual(fit.x)
    allq = motion.joint_pos.copy()
    allq[np.ix_(rows, cols)] = q
    result = refresh_fk(replace(motion, joint_pos=allq), MODEL)
    result.save(out / "motion.npz")
    np.savez_compressed(out / "overlay.npz", rows=rows, cols=cols, joint_pos=q)
    report = {
        "status": "stance_coffee_candidate_pending_exact_room_audit",
        "base": str(src),
        "window_s": [42.3, 43.5],
        "joints": names,
        "root_unchanged": True,
        "solver_success": bool(fit.success),
        "nfev": fit.nfev,
        "max_joint_change_rad": float(abs(q - prior).max()),
        "metrics": metrics,
        "mesh_watertight": bool(mesh.is_watertight),
        "mesh_convex_volume_error_m3": float(abs(abs(mesh.volume) - hull.volume)),
        "limitations": [
            "Convex supporting-plane separation is conservative",
            "All-room100Hz exact audit and support contract pending",
        ],
    }
    (out / "review.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report)


if __name__ == "__main__":
    main()
