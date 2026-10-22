"""Repair only candidate left-arm/table clearance around the inherited 09 seam."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np

from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields, refresh_fk


def main():
    import mujoco
    from scipy.ndimage import gaussian_filter1d
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path)
    ap.add_argument("output", type=Path)
    a = ap.parse_args()
    a.output.mkdir(exist_ok=False)
    for n in [
        "body_prior.npz",
        "human.npz",
        "human_motion_world.npz",
        "source_sync.npz",
        "motion_order.json",
        "lined_rect_basket.npz",
        "table_strawberry.npz",
        "table_strawberry_successor_09.npz",
        "tracks.json",
        "grasp_design_input.json",
        "geometry_config.json",
    ]:
        if (a.source / n).exists():
            shutil.copy2(a.source / n, a.output / n)
    shutil.copy2(__file__, a.output / "repair_connection_source.py")
    model_path = Path(
        "/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof_dex3/g1_29dof_dex3_offsethand.xml"
    )
    m = mujoco.MjModel.from_xml_path(str(model_path))
    d = mujoco.MjData(m)
    motion = MotionFields.load(a.source / "motion.npz", a.source / "motion_order.json")
    q = motion.joint_pos.astype(float).copy()
    geometry = (
        json.loads((a.source / "geometry_config.json").read_text())
        if (a.source / "geometry_config.json").exists()
        else {}
    )
    room_transform = np.array(geometry.get("T_room_correction", np.eye(4)))
    original = q.copy()
    arm = np.array(
        [
            i
            for i, n in enumerate(motion.joint_names)
            if n.startswith("left_") and any(x in n for x in ["shoulder", "elbow", "wrist"])
        ]
    )
    adr = np.array([m.joint(n).qposadr[0] for n in motion.joint_names])
    lo = np.array([m.joint(motion.joint_names[i]).range[0] for i in arm])
    hi = np.array([m.joint(motion.joint_names[i]).range[1] for i in arm])
    samples = []
    for g in range(m.ngeom):
        name = m.geom(g).name or ""
        if not name.startswith("left_") or not any(x in name for x in ["hand", "wrist", "elbow"]):
            continue
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        i = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[i] : m.mesh_vertadr[i] + m.mesh_vertnum[i] : 10].copy()
        samples.append((g, v))
    spec = importlib.util.spec_from_file_location("geo", Path(__file__).parents[1] / "09_lined_rect_basket/fit_hoi.py")
    geo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(geo)
    b = np.load(a.source / "lined_rect_basket.npz")
    br = Rotation.from_quat(b["quaternion_xyzw"]).as_matrix()
    start, end = 370, 411
    for iteration in range(2):
        for k in range(start, end):
            d.qpos[adr] = q[k]
            d.qpos[:3] = motion.body_pos_w[k, 0]
            d.qpos[3:7] = motion.body_quat_w[k, 0]
            target = q[k, arm].copy()
            before = q[k - 1, arm].copy()
            anchor = original[k, arm].copy()

            def residual(x, k=k, target=target, before=before, anchor=anchor):
                d.qpos[adr[arm]] = x
                mujoco.mj_forward(m, d)
                pen = []
                for g, v in samples:
                    w = v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
                    local = (w - room_transform[:3, 3]) @ room_transform[:3, :3]
                    delta = abs(local - np.array([-0.23, -2.544527025, 0.734])) - np.array([0.8, 0.425472975, 0.02])
                    sdf = np.linalg.norm(np.maximum(delta, 0), axis=1) + np.minimum(delta.max(axis=1), 0)
                    pen.append(np.maximum(0.003 - sdf, 0))
                    pen.append(np.maximum(0.003 - geo.basket_sdf((w - b["position_w"][k]) @ br[k]), 0))
                return np.r_[0.1 * (x - anchor), 0.1 * (x - target), 0.05 * (x - before), 50 * np.concatenate(pen)]

            fit = least_squares(
                residual, target, bounds=(np.maximum(lo, anchor - 0.5), np.minimum(hi, anchor + 0.5)), max_nfev=100
            )
            q[k, arm] = fit.x
        if iteration == 0:
            correction = q[:, arm] - original[:, arm]
            correction = gaussian_filter1d(correction, 2, axis=0)
            q[start:end, arm] = original[start:end, arm] + correction[start:end]
    repaired = refresh_fk(replace(motion, joint_pos=q), model_path)
    repaired.save(a.output / "motion.npz")
    delta = abs(q - original)
    changed = np.flatnonzero(delta.max(axis=1) > 1e-8)
    report = {
        "source": str(a.source),
        "scope": "left arm only around table-penetrating 08/09 seam; accepted external 09 files untouched",
        "changed_source_time_s": (90 + changed / 50).tolist(),
        "maximum_joint_change_rad": float(delta.max()),
        "changed_joint_names": [n for i, n in enumerate(motion.joint_names) if delta[:, i].max() > 1e-8],
        "source07_exact": bool(np.array_equal(q[:50], original[:50])),
        "source09_after_98_22_exact": bool(np.array_equal(q[411:], original[411:])),
        "accepted": False,
    }
    (a.output / "connection_repair.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
