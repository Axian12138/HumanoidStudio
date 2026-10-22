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
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--candidate", default="03_04_05_joint_v58")
    p.add_argument("--output", default="03_04_05_joint_v60")
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
    m = mujoco.MjModel.from_xml_path(str(MODEL))
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

    def contact_depth():
        dep = np.zeros(len(pairs))
        for con in d.contact:
            pair = tuple(sorted((int(con.geom1), int(con.geom2))))
            if pair in pm:
                dep[pm[pair]] = max(dep[pm[pair]], -float(con.dist))
        return dep

    records = []
    last = None
    for k in range(round((39.2 - 27) * 50), round((41.3 - 27) * 50) + 1):
        t = 27 + k / 50
        d.qpos[:3] = motion.body_pos_w[k, 0]
        d.qpos[3:7] = motion.body_quat_w[k, 0]
        d.qpos[addr] = baseline.joint_pos[k]
        mujoco.mj_forward(m, d)
        limit = contact_depth() + 0.0005
        d.qpos[addr] = q[k]
        targetp = motion.body_pos_w[k, wi]
        targetr = Rotation.from_quat(motion.body_quat_w[k, wi][[1, 2, 3, 0]])
        prior = q[k, cols].copy()
        if last is None:
            last = prior.copy()

        def residual(x, targetp=targetp, targetr=targetr, limit=limit, prior=prior, last=last):
            d.qpos[addr[cols]] = x
            mujoco.mj_forward(m, d)
            rot = Rotation.from_matrix(d.xmat[w].reshape(3, 3))
            return np.r_[
                (d.xpos[w] - targetp) * 2000,
                (rot * targetr.inv()).as_rotvec() * 300,
                np.maximum(contact_depth() - limit, 0) * 3000,
                (x - prior) * 1.0,
                (x - last) * 0.2,
            ]

        fit = least_squares(residual, np.clip(prior, lo + 1e-7, hi - 1e-7), bounds=(lo, hi), max_nfev=120, ftol=1e-8)
        residual(fit.x)
        q[k, cols] = fit.x
        last = fit.x.copy()
        records.append(
            {
                "source_s": t,
                "wrist_error_m": float(np.linalg.norm(d.xpos[w] - targetp)),
                "self_excess_m": float(np.maximum(contact_depth() - limit, 0).max()),
                "q_delta_max": float(abs(fit.x - prior).max()),
            }
        )
        if k % 25 == 0:
            print(records[-1], flush=True)
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
                "window": [39.2, 41.3],
                "records": records,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
