"""Continuous deep-hook path repair, preserving the seated palm and arm branch."""

import argparse
import json
import xml.etree.ElementTree as ET
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
    p.add_argument("--only-side", choices=["left", "right"])
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    parent = json.load(open(a.run / "inputs/parent.json"))
    info = parent["parent"]
    s = MotionFields.load(a.motion, info["order"])
    q = s.joint_pos.copy()
    waist = s.joint_names.index("waist_pitch_joint")
    q[:, waist] = np.clip(q[:, waist], -0.52, 0.52)
    robot = Path(
        "/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof_dex3/g1_29dof_dex3_offsethand.xml"
    )
    tree = ET.parse(robot)
    root = tree.getroot()
    root.find("compiler").set("meshdir", str(robot.parent / "meshes"))
    tracks = {}
    ognames = []
    for n in ["flower_bouquet", "small_wicker_basket"]:
        tracks[n] = np.load(Path(info["tracks"]) / (n + ".npz"))
        body = ET.SubElement(root.find("worldbody"), "body", name="prop_" + n)
        ET.SubElement(body, "freejoint", name="prop_" + n + "_free")
        for j, f in enumerate(sorted(Path(info["assets"]).glob(n + "_hull*.obj"))):
            name = n + str(j)
            ET.SubElement(root.find("asset"), "mesh", name=name, file=str(f))
            ET.SubElement(body, "geom", name=name, type="mesh", mesh=name, mass=".01")
            ognames.append(name)
    m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    d = mujoco.MjData(m)
    ad = np.array([m.joint(n).qposadr[0] for n in s.joint_names])
    objad = {n: m.joint("prop_" + n + "_free").qposadr[0] for n in tracks}
    og = [m.geom(n).id for n in ognames]
    reports = []
    for side, begin, end in [("right", 123.3, 124.6)]:
        if a.only_side and side != a.only_side:
            continue
        cols = np.array(
            [
                s.joint_names.index(side + "_" + n + "_joint")
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
        hg = [
            m.geom(side + "_hand_" + part + "_collision").id
            for part in ["palm", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1"]
        ]
        for part in ["index_0", "middle_0"]:
            proxy = m.geom(side + "_hand_" + part + "_collision").id
            body = m.geom_bodyid[proxy]
            visual = next(
                i for i in range(m.ngeom) if m.geom_bodyid[i] == body and m.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH
            )
            hg[hg.index(proxy)] = visual
        pairs = [(g, h) for g in hg for h in og]
        pairs += [
            (m.geom(side + "_shoulder_yaw_collision").id, m.geom("torso_collision" + str(j)).id) for j in [1, 2, 3]
        ]
        pairs += [(m.geom(side + "_elbow_yaw_collision").id, m.geom("torso_collision" + str(j)).id) for j in [1, 2, 3]]
        pairs += [(m.geom(side + "_wrist_collision").id, m.geom(side + "_hip_collision").id)]
        first, last = [round((t - 108) * 50) for t in [begin, end]]
        ix = np.arange(first, last + 1)
        size = len(ix) - 4
        full = q[np.ix_(ix, cols)].astype(float)
        base = full.copy()
        roots = []
        tp = []
        tr = []
        for k in ix[2:-2]:
            d.qpos[:3] = s.body_pos_w[k, 0]
            d.qpos[3:7] = s.body_quat_w[k, 0]
            d.qpos[ad] = q[k]
            for n, v in tracks.items():
                aa = objad[n]
                d.qpos[aa : aa + 3] = v["position_w"][k]
                d.qpos[aa + 3 : aa + 7] = v["quaternion_xyzw"][k, [3, 0, 1, 2]]
            mujoco.mj_forward(m, d)
            roots.append(d.qpos.copy())
            tp.append(d.xpos[wb].copy())
            tr.append(d.xmat[wb].reshape(3, 3).copy())
        npose = 6 + 2 * len(pairs)
        times = 108 + ix[2:-2] / 50
        held = (times >= 117.7) & (times <= 120.0)
        ps = np.where(held, 0.0005, 0.02)
        rs = np.where(held, 0.005, 0.15)

        def fun(
            x, full=full, size=size, roots=roots, tp=tp, tr=tr, cols=cols, wb=wb, pairs=pairs, base=base, ps=ps, rs=rs
        ):
            full[2:-2] = x.reshape(size, 7)
            rows = []
            for j in range(size):
                d.qpos[:] = roots[j]
                d.qpos[ad[cols]] = full[j + 2]
                mujoco.mj_forward(m, d)
                gap = np.array([mujoco.mj_geomDistance(m, d, g, h, 0.02, None) for g, h in pairs])
                d.qpos[:] = (roots[j] + roots[max(0, j - 1)]) / 2
                d.qpos[ad[cols]] = (full[j + 1] + full[j + 2]) / 2
                mujoco.mj_forward(m, d)
                midgap = np.array([mujoco.mj_geomDistance(m, d, g, h, 0.02, None) for g, h in pairs])
                d.qpos[:] = roots[j]
                d.qpos[ad[cols]] = full[j + 2]
                mujoco.mj_forward(m, d)
                rows.extend(
                    np.r_[
                        (d.xpos[wb] - tp[j]) / ps[j],
                        R.from_matrix(d.xmat[wb].reshape(3, 3) @ tr[j].T).as_rotvec() / rs[j],
                        np.minimum(gap - 0.0002, 0) / 0.0001,
                        np.minimum(midgap - 0.0002, 0) / 0.0001,
                    ]
                )
            return np.r_[
                rows,
                ((full[2:-2] - base[2:-2]) / 0.5).ravel(),
                (np.diff(full, n=2, axis=0) * 2500 / 80).ravel(),
                (np.maximum(abs(np.diff(full, axis=0)) * 50 - 6, 0) / 0.02).ravel(),
            ]

        total = size * npose + size * 7 + (size + 2) * 7 + (size + 3) * 7
        sp = lil_matrix((total, size * 7), dtype=int)
        for j in range(size):
            sp[j * npose : (j + 1) * npose, max(0, j - 1) * 7 : (j + 1) * 7] = 1
            for c in range(7):
                col = j * 7 + c
                sp[size * npose + col, col] = 1
                off = size * (npose + 7)
                for z in [j, j + 1, j + 2]:
                    sp[off + z * 7 + c, col] = 1
                off += (size + 2) * 7
                for z in [j + 1, j + 2]:
                    sp[off + z * 7 + c, col] = 1
        lim = np.array([m.joint(s.joint_names[c]).range for c in cols])
        lim[5] = [-0.47, 0.47]
        lim[6] = [-0.65, 0.65]
        result = least_squares(
            fun,
            np.clip(base[2:-2], lim[:, 0] + 1e-7, lim[:, 1] - 1e-7).ravel(),
            bounds=(np.tile(lim[:, 0], size), np.tile(lim[:, 1], size)),
            jac_sparsity=sp.tocsr(),
            max_nfev=60,
            ftol=1e-6,
        )
        full[2:-2] = result.x.reshape(size, 7)
        q[np.ix_(ix, cols)] = full
        row = {"side": side, "cost": result.cost, "nfev": result.nfev, "success": bool(result.success)}
        reports.append(row)
        print(row, flush=True)
    refresh_fk(replace(s, joint_pos=q), robot).save(a.out / "motion.npz")
    (a.out / "motion_order.json").write_bytes(Path(info["order"]).read_bytes())
    (a.out / "report.json").write_text(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
