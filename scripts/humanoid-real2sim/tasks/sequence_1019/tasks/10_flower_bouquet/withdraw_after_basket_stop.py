"""Keep the pushing hand outside a stopped basket and preserve its withdrawal."""

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
    from scipy.spatial.transform import Slerp

    from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields, refresh_fk

    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    parent = json.load(open(a.run / "inputs/parent.json"))
    info = parent["parent"]
    s = MotionFields.load(parent["motion"], info["order"])
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
    wb = m.body("left_wrist_yaw_link").id
    cols = [
        s.joint_names.index("left_" + n + "_joint")
        for n in ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"]
    ]
    oldbasket = np.load(Path(parent["original_tracks"]) / "small_wicker_basket.npz")
    hand = [
        m.geom("left_hand_" + n + "_collision").id
        for n in ["palm", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1"]
    ]
    for part in ["index_0", "middle_0"]:
        g = m.geom("left_hand_" + part + "_collision").id
        hand[hand.index(g)] = next(
            i
            for i in range(m.ngeom)
            if m.geom_bodyid[i] == m.geom_bodyid[g] and m.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH
        )
    pairs = [(g, h) for g in hand for h in og]
    pairs += [
        (m.geom("left_" + part + "_collision").id, m.geom("torso_collision" + str(j)).id)
        for part in ["shoulder_yaw", "elbow_yaw"]
        for j in [1, 2, 3]
    ]
    lim = np.array([m.joint(s.joint_names[c]).range for c in cols])
    previous = q[945, cols].astype(float)
    rows = []
    for k in range(945, 1036):
        t = 108 + k / 50
        d.qpos[:3] = s.body_pos_w[k, 0]
        d.qpos[3:7] = s.body_quat_w[k, 0]
        d.qpos[ad] = s.joint_pos[k]
        for n, v in tracks.items():
            aa = oa[n]
            d.qpos[aa : aa + 3] = v["position_w"][k]
            d.qpos[aa + 3 : aa + 7] = v["quaternion_xyzw"][k, [3, 0, 1, 2]]
        mujoco.mj_forward(m, d)
        wp = d.xpos[wb].copy()
        wr = d.xmat[wb].reshape(3, 3).copy()
        br = R.from_quat(tracks["small_wicker_basket"]["quaternion_xyzw"][k]).as_matrix()
        oldr = R.from_quat(oldbasket["quaternion_xyzw"][k]).as_matrix()
        delta = br @ oldr.T
        w = 1 - smooth((t - 127.8) / 0.9)
        target = (
            w * (tracks["small_wicker_basket"]["position_w"][k] + delta @ (wp - oldbasket["position_w"][k]))
            + (1 - w) * wp
        )
        rot = Slerp([0, 1], R.from_matrix([wr, delta @ wr]))([w]).as_matrix()[0]
        base = q[k, cols].astype(float)

        def fun(x, target=target, rot=rot, base=base, previous=previous):
            d.qpos[ad[cols]] = x
            mujoco.mj_forward(m, d)
            gaps = np.array([mujoco.mj_geomDistance(m, d, g, h, 0.03, None) for g, h in pairs])
            return np.r_[
                (d.xpos[wb] - target) * 1000,
                R.from_matrix(d.xmat[wb].reshape(3, 3) @ rot.T).as_rotvec() * 150,
                np.minimum(gaps - 0.0005, 0) * 8000,
                (x - base) * 5,
                (x - previous) * 2,
            ]

        fit = least_squares(
            fun,
            np.clip(base, lim[:, 0] + 1e-8, lim[:, 1] - 1e-8),
            bounds=(lim[:, 0], lim[:, 1]),
            max_nfev=100,
            diff_step=1e-4,
            ftol=1e-7,
        )
        q[k, cols] = fit.x
        previous = fit.x
        res = fun(fit.x)
        rows.append({"time_s": t, "position_error_m": float(np.linalg.norm(res[:3]) / 1000), "cost": float(fit.cost)})
    q[945, cols] = s.joint_pos[945, cols]
    q[1035, cols] = s.joint_pos[1035, cols]
    refresh_fk(replace(s, joint_pos=q), robot).save(a.out / "motion.npz")
    (a.out / "motion_order.json").write_bytes(Path(info["order"]).read_bytes())
    (a.out / "report.json").write_text(json.dumps(rows, indent=2))
    print("max error", max(x["position_error_m"] for x in rows))


if __name__ == "__main__":
    main()
