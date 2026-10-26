"""Continuous arm-window fit with fixed original boundaries and explicit speed penalties."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np


def main():
    import mujoco
    from scipy.optimize import least_squares
    from scipy.sparse import lil_matrix
    from scipy.spatial.transform import Rotation as R

    from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields, refresh_fk

    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--motion", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--nfev", type=int, default=100)
    p.add_argument("--only-side", choices=["left", "right"])
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    parent = json.load(open(a.run / "inputs/parent.json"))
    order = parent["parent"]["order"]
    seed = MotionFields.load(a.motion, order)
    original = MotionFields.load(parent["motion"], order)
    robot = Path(
        "/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof_dex3/g1_29dof_dex3_offsethand.xml"
    )
    m = mujoco.MjModel.from_xml_path(str(robot))
    d = mujoco.MjData(m)
    m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    addr = np.array([m.joint(n).qposadr[0] for n in seed.joint_names])
    q = seed.joint_pos.copy()
    reports = []
    for side, begin, end, contact, release in [
        ("right", 110.8, 126.0, 113.2, 123.0),
        ("left", 116.0, 122.4, 118.7, 120.4),
    ]:
        if a.only_side and side != a.only_side:
            continue
        cols = np.array(
            [
                seed.joint_names.index(side + "_" + n + "_joint")
                for n in [
                    "shoulder_pitch",
                    "shoulder_roll",
                    "shoulder_yaw",
                    "elbow",
                    "wrist_roll",
                    "wrist_pitch",
                    "wrist_yaw",
                ]
            ]
        )
        wb = m.body(side + "_wrist_yaw_link").id
        first, last = [round((t - 108) * 50) for t in [begin, end]]
        ix = np.arange(first, last + 1)
        size = len(ix) - 4
        full = seed.joint_pos[np.ix_(ix, cols)].astype(float)
        full[:2] = original.joint_pos[np.ix_(ix[:2], cols)]
        full[-2:] = original.joint_pos[np.ix_(ix[-2:], cols)]
        base = full.copy()
        pairs = [
            (m.geom(side + "_shoulder_yaw_collision").id, m.geom("torso_collision" + str(i)).id) for i in [1, 2, 3]
        ] + [
            (m.geom(side + "_elbow_yaw_collision").id, m.geom("torso_collision2").id),
            (m.geom(side + "_elbow_yaw_collision").id, m.geom(side + "_wrist_collision").id),
        ]
        roots = []
        tp = []
        tr = []
        tg = []
        for k in ix[2:-2]:
            d.qpos[:3] = seed.body_pos_w[k, 0]
            d.qpos[3:7] = seed.body_quat_w[k, 0]
            d.qpos[addr] = seed.joint_pos[k]
            mujoco.mj_forward(m, d)
            roots.append(d.qpos.copy())
            tp.append(d.xpos[wb].copy())
            tr.append(d.xmat[wb].reshape(3, 3).copy())
            tg.append([mujoco.mj_geomDistance(m, d, g, h, 0.02, None) for g, h in pairs])
        tp = np.array(tp)
        tr = np.array(tr)
        tg = np.minimum(tg, 0.001)
        times = 108 + ix[2:-2] / 50
        held = (times >= contact) & (times <= release)
        ps = np.where(held, 0.0008, 0.012)
        rs = np.where(held, 0.008, 0.10)
        npose = 6 + len(pairs)

        def fun(
            x,
            full=full,
            size=size,
            cols=cols,
            roots=roots,
            tp=tp,
            tr=tr,
            tg=tg,
            wb=wb,
            pairs=pairs,
            ps=ps,
            rs=rs,
            base=base,
        ):
            full[2:-2] = x.reshape(size, 7)
            rows = []
            for j in range(size):
                d.qpos[:] = roots[j]
                d.qpos[addr[cols]] = full[j + 2]
                mujoco.mj_forward(m, d)
                gaps = np.array([mujoco.mj_geomDistance(m, d, g, h, 0.02, None) for g, h in pairs])
                rows.extend(
                    np.r_[
                        (d.xpos[wb] - tp[j]) / ps[j],
                        R.from_matrix(d.xmat[wb].reshape(3, 3) @ tr[j].T).as_rotvec() / rs[j],
                        np.minimum(gaps - tg[j], 0) / 0.0001,
                    ]
                )
            return np.r_[
                rows,
                ((full[2:-2] - base[2:-2]) / 1.0).ravel(),
                (np.diff(full, n=2, axis=0) * 2500 / 45).ravel(),
                (np.maximum(abs(np.diff(full, axis=0)) * 50 - 5.8, 0) / 0.01).ravel(),
            ]

        total = size * npose + size * 7 + (size + 2) * 7 + (size + 3) * 7
        sp = lil_matrix((total, size * 7), dtype=int)
        for j in range(size):
            sp[j * npose : (j + 1) * npose, j * 7 : (j + 1) * 7] = 1
            for c in range(7):
                col = j * 7 + c
                sp[size * npose + col, col] = 1
                off = size * (npose + 7)
                for z in [j, j + 1, j + 2]:
                    sp[off + z * 7 + c, col] = 1
                off += (size + 2) * 7
                for z in [j + 1, j + 2]:
                    sp[off + z * 7 + c, col] = 1
        limits = np.array([m.joint(seed.joint_names[c]).range for c in cols])
        lo = np.tile(limits[:, 0], (size, 1))
        hi = np.tile(limits[:, 1], (size, 1))
        lo[held, 5] = -0.44
        hi[held, 5] = 0.44
        result = least_squares(
            fun,
            np.clip(base[2:-2], lo + 1e-7, hi - 1e-7).ravel(),
            bounds=(lo.ravel(), hi.ravel()),
            jac_sparsity=sp.tocsr(),
            max_nfev=a.nfev,
            ftol=1e-6,
        )
        full[2:-2] = result.x.reshape(size, 7)
        q[np.ix_(ix, cols)] = full
        report = {
            "side": side,
            "success": bool(result.success),
            "nfev": result.nfev,
            "cost": result.cost,
            "speed": float(abs(np.diff(full, axis=0) * 50).max()),
        }
        reports.append(report)
        print(report, flush=True)
    refresh_fk(replace(seed, joint_pos=q), robot).save(a.out / "motion.npz")
    (a.out / "report.json").write_text(json.dumps(reports, indent=2))
    (a.out / "motion_order.json").write_bytes(Path(order).read_bytes())


if __name__ == "__main__":
    main()
