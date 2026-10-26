"""Redistribute right wrist rotation to shoulder/elbow while retaining the web axis."""

import argparse
import json
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path

import numpy as np


def smooth(x):
    x = np.clip(x, 0, 1)
    return np.clip(x**3 * (10 - 15 * x + 6 * x * x), 0, 1)


def main():
    import mujoco
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation as R

    from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields, refresh_fk

    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--motion", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    parent = json.load(open(a.run / "inputs/parent.json"))
    info = parent["parent"]
    s = MotionFields.load(parent["motion"], info["order"])
    reference = s
    s = MotionFields.load(a.motion, info["order"])
    q = s.joint_pos.copy()
    robot = Path(
        "/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof_dex3/g1_29dof_dex3_offsethand.xml"
    )
    root = ET.parse(robot).getroot()
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
    oa = {n: m.joint("prop_" + n + "_free").qposadr[0] for n in tracks}
    og = [m.geom(n).id for n in ognames]
    wb = m.body("right_wrist_yaw_link").id
    elbow = m.body("right_elbow_link").id
    shoulder = m.body("right_shoulder_roll_link").id
    cols = [
        s.joint_names.index("right_" + n + "_joint")
        for n in ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"]
    ]
    center = np.array([0.1243705897805318, 0.05823026969988426, 0])
    hand = [
        m.geom("right_hand_" + part + "_collision").id
        for part in ["palm", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1"]
    ]
    for part in ["index_0", "middle_0"]:
        g = m.geom("right_hand_" + part + "_collision").id
        hand[hand.index(g)] = next(
            i
            for i in range(m.ngeom)
            if m.geom_bodyid[i] == m.geom_bodyid[g] and m.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH
        )
    pairs = [(g, h) for g in hand for h in og]
    pairs += [
        (m.geom("right_" + part + "_collision").id, m.geom("torso_collision" + str(j)).id)
        for part in ["shoulder_yaw", "elbow_yaw"]
        for j in [1, 2, 3]
    ]
    bounds = np.array([m.joint(s.joint_names[c]).range for c in cols])
    branches = json.load(open(a.run / "review/right_axis_negative.json"))
    bt = np.array([v["t"] for v in branches])
    bq = np.array([v["q"] for v in branches])

    def setpose(k, values):
        d.qpos[:3] = s.body_pos_w[k, 0]
        d.qpos[3:7] = s.body_quat_w[k, 0]
        d.qpos[ad] = values
        for n, v in tracks.items():
            aa = oa[n]
            d.qpos[aa : aa + 3] = v["position_w"][k]
            d.qpos[aa + 3 : aa + 7] = v["quaternion_xyzw"][k, [3, 0, 1, 2]]
        mujoco.mj_forward(m, d)

    first, last = 100, 840
    previous = q[first, cols].astype(float)
    rows = []
    for k in range(first, last + 1):
        t = 108 + k / 50
        setpose(k, reference.joint_pos[k])
        oldr = d.xmat[wb].reshape(3, 3).copy()
        target = d.xpos[wb] + oldr @ center
        axis = oldr[:, 2]
        setpose(k, q[k])
        weight = min(smooth((t - 110) / 2), smooth((124.8 - t) / 2.8))
        anchor = np.array([np.interp(t, bt, bq[:, c]) for c in range(7)])
        posture = (1 - weight) * reference.joint_pos[k, cols] + weight * anchor
        lim = bounds.copy()
        for c, limit in [(4, 1.6), (5, 0.47), (6, 0.65)]:
            blended_limit = (1 - weight) * max(limit, abs(float(reference.joint_pos[k, cols[c]]))) + weight * limit
            lim[c] = [max(bounds[c, 0], -blended_limit), min(bounds[c, 1], blended_limit)]

        def fun(x, target=target, axis=axis, oldr=oldr, weight=weight, previous=previous, posture=posture):
            d.qpos[ad[cols]] = x
            mujoco.mj_forward(m, d)
            wr = d.xmat[wb].reshape(3, 3)
            gaps = np.array([mujoco.mj_geomDistance(m, d, g, h, 0.02, None) for g, h in pairs])
            return np.r_[
                (d.xpos[wb] + wr @ center - target) * 3000,
                (wr[:, 2] - axis) * 500,
                R.from_matrix(wr @ oldr.T).as_rotvec() * (1 - weight) * 60,
                np.minimum(gaps - 0.0002, 0) * 5000,
                (x - previous) * 2,
                (x - posture) * 0.8,
                x[5:] * 3 * weight,
                x[4] * 0.6 * weight,
                max(0, d.xpos[elbow][2] - d.xpos[shoulder][2] + 0.03) * 20 * weight,
            ]

        ini = np.clip(previous, lim[:, 0] + 1e-8, lim[:, 1] - 1e-8)
        fit = least_squares(fun, ini, bounds=(lim[:, 0], lim[:, 1]), max_nfev=75, diff_step=1e-4, ftol=1e-6)
        q[k, cols] = fit.x
        previous = fit.x
        res = fun(fit.x)
        row = {
            "t": t,
            "position_error_m": float(np.linalg.norm(res[:3]) / 3000),
            "axis_error": float(np.linalg.norm(res[3:6]) / 500),
            "yaw": float(fit.x[6]),
            "pitch": float(fit.x[5]),
            "cost": float(fit.cost),
        }
        rows.append(row)
        if k % 50 == 0:
            print(row, flush=True)
    q[first, cols] = s.joint_pos[first, cols]
    q[last, cols] = s.joint_pos[last, cols]
    refresh_fk(replace(s, joint_pos=q), robot).save(a.out / "motion.npz")
    (a.out / "motion_order.json").write_bytes(Path(info["order"]).read_bytes())
    (a.out / "report.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
