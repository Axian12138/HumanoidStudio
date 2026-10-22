"""Local drawer timing diagnostic using source wrist progression and visual closure."""

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
    from scipy.interpolate import PchipInterpolator
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--base", default="03_04_05_joint_v76")
    p.add_argument("--output", default="05_drawer_event_timing/v1")
    p.add_argument("--contact-geometry", action="store_true")
    p.add_argument("--actual-hand-surface", action="store_true")
    p.add_argument("--extent", type=float, default=0.19)
    p.add_argument("--model", type=Path)
    a = p.parse_args()
    src = a.run / "retarget/candidates" / a.base
    out = a.run / "repairs" / a.output
    out.mkdir(parents=True, exist_ok=False)
    motion = MotionFields.load(src / "motion.npz", src / "motion_order.json")
    human = np.load(src / "human_motion_world.npz")
    hwi = list(human["joint_names"]).index("right_wrist")
    t = 27 + np.arange(len(motion.joint_pos)) / 50
    old = np.load(src / "drawer_slide.npz")["position"]
    # Source closure is visually bracketed at Header 53.272--53.472 s.
    # This diagnostic picks its midpoint; contact onset remains visually occluded.
    start, end = 51.0, 53.38
    knots = np.r_[np.arange(start, end, 0.1), end]
    hy = np.interp(knots, t, human["positions_w"][:, hwi, 1])
    progress = np.maximum.accumulate(hy[0] - hy)
    progress /= progress[-1]
    new = a.extent * (1 - PchipInterpolator(knots, progress)(np.clip(t, start, end)))
    new[t <= start] = a.extent
    new[t >= end] = 0
    m = mujoco.MjModel.from_xml_path(
        str(a.model or a.run / "scene/03_04_05/robot_props_drawer.xml") if a.contact_geometry else str(MODEL)
    )
    d = mujoco.MjData(m)
    arm = [
        "right_" + n + "_joint"
        for n in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
    ]
    cols = [motion.joint_names.index(n) for n in arm]
    addr = np.array([m.joint(n).qposadr[0] for n in motion.joint_names])
    lo = np.array([m.joint(n).range[0] for n in arm])
    hi = np.array([m.joint(n).range[1] for n in arm])
    wb = m.body("right_wrist_yaw_link").id
    if a.contact_geometry:
        da = m.joint("drawer_slide").qposadr[0]
        db = m.body("drawer").id
        dg = [g for g in range(m.ngeom) if m.geom_bodyid[g] == db and m.geom_contype[g]]
        hg = [
            g
            for g in range(m.ngeom)
            if m.body(int(m.geom_bodyid[g])).name.startswith("right_hand") and m.geom_contype[g]
        ]
        visual = []
        if a.actual_hand_surface:
            for g in range(m.ngeom):
                if (
                    not m.body(int(m.geom_bodyid[g])).name.startswith("right_hand")
                    or m.geom_contype[g]
                    or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH
                ):
                    continue
                mi = int(m.geom_dataid[g])
                va, vn = int(m.mesh_vertadr[mi]), int(m.mesh_vertnum[mi])
                fa, fn = int(m.mesh_faceadr[mi]), int(m.mesh_facenum[mi])
                verts = m.mesh_vert[va : va + vn]
                tri = verts[m.mesh_face[fa : fa + fn]]
                samples = np.vstack(
                    (
                        verts,
                        tri.mean(axis=1),
                        (tri[:, 0] + tri[:, 1]) / 2,
                        (tri[:, 1] + tri[:, 2]) / 2,
                        (tri[:, 2] + tri[:, 0]) / 2,
                    )
                )
                # Only the optimizer uses the voxel subset; independent checks sample
                # every original triangle at <=2mm spacing.
                _, keep = np.unique(np.round(samples / 0.001), axis=0, return_index=True)
                samples = samples[keep]
                visual.append((g, samples))
            print({"optimizer_visual_surface_samples": sum(len(v) for _, v in visual)}, flush=True)
    q = motion.joint_pos.astype(float).copy()
    window_start = 49.8 if a.extent != 0.19 else start
    rows = np.flatnonzero((t >= window_start) & (t <= 54))
    records = []
    previous = np.zeros(7)
    for k in rows:
        baseline = q[k, cols].copy()
        d.qpos[:3] = motion.body_pos_w[k, 0]
        d.qpos[3:7] = motion.body_quat_w[k, 0]
        d.qpos[addr] = q[k]
        if a.contact_geometry:
            d.qpos[da] = new[k]
        mujoco.mj_forward(m, d)
        blend = np.clip((t[k] - window_start) / max(start - window_start, 1e-6), 0, 1)
        blend = blend**3 * (10 - 15 * blend + 6 * blend**2)
        target = d.xpos[wb].copy() + np.array([0, (new[k] - old[k]) * blend, 0])
        rot = Rotation.from_matrix(d.xmat[wb].reshape(3, 3))

        def residual(x, k=k, target=target, rot=rot, baseline=baseline, previous=previous):
            d.qpos[addr[cols]] = x
            mujoco.mj_forward(m, d)
            contact = np.zeros(1)
            if a.contact_geometry:
                if a.actual_hand_surface:
                    distances = []
                    for g, samples in visual:
                        world = samples @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
                        minimum = 0.3
                        for box in dg:
                            local = (world - d.geom_xpos[box]) @ d.geom_xmat[box].reshape(3, 3)
                            delta = abs(local) - m.geom_size[box]
                            sdf = np.linalg.norm(np.maximum(delta, 0), axis=1) + np.minimum(delta.max(axis=1), 0)
                            minimum = min(minimum, float(sdf.min()))
                        distances.append(minimum)
                    distances = np.array(distances)
                else:
                    distances = np.array([min(mujoco.mj_geomDistance(m, d, h, g, 0.3, None) for g in dg) for h in hg])
                fade = min(np.clip((t[k] - start) / 0.15, 0, 1), np.clip((54 - t[k]) / 0.3, 0, 1))
                contact = np.r_[
                    max(distances.min() - 0.0005, 0) * 3000 * fade,
                    np.minimum(distances + (0.0025 if a.actual_hand_surface else 0.004), 0) * 5000,
                ]
            return np.r_[
                (d.xpos[wb] - target) * (300 if a.contact_geometry else 3000),
                (Rotation.from_matrix(d.xmat[wb].reshape(3, 3)) * rot.inv()).as_rotvec() * 1000,
                (x - baseline) * 0.3,
                (x - baseline - previous) * 0.5,
                contact,
            ]

        fit = least_squares(residual, np.clip(baseline + previous, lo + 1e-7, hi - 1e-7), bounds=(lo, hi), max_nfev=60)
        if k in (rows[0], rows[-1]):
            fit.x = baseline
        error = residual(fit.x)
        q[k, cols] = fit.x
        previous = fit.x - baseline
        records.append(
            {
                "source_s": float(t[k]),
                "wrist_error_m": float(np.linalg.norm(error[:3]) / (300 if a.contact_geometry else 3000)),
            }
        )
        if k % 10 == 0:
            print({"source_s": float(t[k]), "wrist_error_m": records[-1]["wrist_error_m"]}, flush=True)
    result = refresh_fk(replace(motion, joint_pos=q), MODEL)
    for f in src.iterdir():
        if (
            f.is_file()
            and f.suffix in (".npz", ".json")
            and f.name not in ("motion.npz", "drawer_slide.npz", "review.json", "manifest.json")
        ):
            shutil.copy2(f, out / f.name)
    result.save(out / "motion.npz")
    np.savez_compressed(out / "drawer_slide.npz", position=new, fps=50, source_time_s=t)
    np.savez_compressed(
        out / "overlay.npz",
        rows=rows,
        columns=cols,
        joint_pos=q[np.ix_(rows, cols)],
        baseline_joint_pos=motion.joint_pos[np.ix_(rows, cols)],
        drawer_position=new,
        baseline_drawer_position=old,
    )
    report = {
        "status": "timing_diagnostic_pending_room_contact_source_acceptance",
        "base": str(src),
        "closure_visual_header_bracket_s": [53.272338944, 53.472338824],
        "closure_candidate_s": end,
        "drawer_extent_m": a.extent,
        "push_start_candidate_s": start,
        "method": f"Normalize source human wrist forward progression to {a.extent:.9f}m drawer travel; preserve wrist orientation and approximate wrist/drawer relative offset with contact constraints.",
        "limitations": [
            "Travel is provided by --extent; it is not measured from the wrist excursion. Source edge fit must be reviewed separately.",
            "Contact onset is occluded; 51.0s remains a diagnostic estimate.",
            "No room or actual hand/drawer collision checks are implied by IK residual.",
        ],
        "joints": arm,
        "authored_contact_geometry_constrained": a.contact_geometry,
        "actual_hand_surface_constrained": a.actual_hand_surface,
        "max_joint_change_rad": float(abs(q - motion.joint_pos).max()),
        "max_wrist_error_m": max(x["wrist_error_m"] for x in records),
        "records": records,
    }
    (out / "review.json").write_text(json.dumps(report, indent=2) + "\n")
    print({k: v for k, v in report.items() if k != "records"}, flush=True)


if __name__ == "__main__":
    main()
