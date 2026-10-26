"""Enter an open palm over the rim, curl both fingers deeply, open before withdrawing."""

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
    design = json.load(open(a.run / "review/grip_design.json"))[0]
    fc = [s.joint_names.index("left_hand_" + n + "_joint") for n in design["finger_names"]]
    closed = np.array(design["finger_angles"])
    opened = np.array([0, 0.2, 0.3, 0, 0, 0, 0])
    rimlocal = np.array(design["rim_in_wrist"])
    anchor = np.array(
        next(x for x in json.load(open(a.run / "review/branch_comparison.json")) if x["branch"] == "positive_elbow")[
            "q"
        ]
    )
    lim = np.array([m.joint(s.joint_names[c]).range for c in cols])
    lim[5] = [-0.44, 0.44]
    hand = []
    for part in ["palm", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1"]:
        g = m.geom("left_hand_" + part + "_collision").id
        b = m.geom_bodyid[g]
        hand.extend(
            [g] + [i for i in range(m.ngeom) if m.geom_bodyid[i] == b and m.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH]
        )
    hand = list(set(hand))
    pairs = [(g, h) for g in hand for h in og]
    pairs += [
        (m.geom("left_" + part + "_collision").id, m.geom("torso_collision" + str(i)).id)
        for part in ["shoulder_yaw", "elbow_yaw"]
        for i in [1, 2, 3]
    ]

    def setpose(k):
        d.qpos[:3] = s.body_pos_w[k, 0]
        d.qpos[3:7] = s.body_quat_w[k, 0]
        d.qpos[ad] = q[k]
        for n, v in tracks.items():
            aa = oa[n]
            d.qpos[aa : aa + 3] = v["position_w"][k]
            d.qpos[aa + 3 : aa + 7] = v["quaternion_xyzw"][k, [3, 0, 1, 2]]
        mujoco.mj_forward(m, d)

    first, last = 400, 740
    setpose(first)
    startp = d.xpos[wb].copy()
    startr = d.xmat[wb].reshape(3, 3).copy()
    setpose(last)
    endp = d.xpos[wb].copy()
    endr = d.xmat[wb].reshape(3, 3).copy()
    previous = s.joint_pos[first, cols].astype(float)
    rows = []
    targets = []
    for k in range(first, last + 1):
        t = 108 + k / 50
        b = tracks["small_wicker_basket"]
        br = R.from_quat(b["quaternion_xyzw"][k]).as_matrix()
        wr = br[:, [1, 2, 0]]
        rim = b["position_w"][k] + br @ np.array([0, -0.16872634, 0.20841585])
        grasp = rim - wr @ rimlocal
        hover = grasp + br @ np.array([0, -0.08, 0.12])
        if t < 117.0:
            u = smooth((t - 116) / 1.0)
            target = (1 - u) * startp + u * hover
            rot = Slerp([0, 1], R.from_matrix([startr, wr]))([u]).as_matrix()[0]
        elif t < 117.7:
            u = smooth((t - 117.0) / 0.7)
            target = (1 - u) * hover + u * grasp
            rot = wr
        elif t <= 120.4:
            target = grasp
            rot = wr
        elif t < 120.85:
            u = smooth((t - 120.4) / 0.45)
            target = grasp + np.array([0, 0, 0.12 * u])
            rot = wr
        elif t < 121.2:
            u = smooth((t - 120.85) / 0.35)
            target = grasp + br @ np.array([0, -0.08 * u, 0.12])
            rot = wr
        else:
            u = smooth((t - 121.2) / 1.6)
            target = (1 - u) * hover + u * endp
            rot = Slerp([0, 1], R.from_matrix([wr, endr]))([u]).as_matrix()[0]
        if t < 117.0:
            handq = (1 - smooth(t - 116)) * s.joint_pos[first, fc] + smooth(t - 116) * opened
        elif t < 117.7:
            handq = opened
        elif t <= 120.0:
            handq = (1 - smooth((t - 117.7) / 1.0)) * opened + smooth((t - 117.7) / 1.0) * closed
        elif t <= 120.85:
            handq = (1 - smooth((t - 120.0) / 0.85)) * closed + smooth((t - 120.0) / 0.85) * opened
        else:
            handq = (1 - smooth((t - 121.7) / 1.1)) * opened + smooth((t - 121.7) / 1.1) * s.joint_pos[last, fc]
        q[k, fc] = handq
        setpose(k)
        weight = min(smooth((t - 116) / 1.0), smooth((122.8 - t) / 1.1))
        posture = (1 - weight) * s.joint_pos[k, cols] + weight * anchor

        def fun(x, target=target, rot=rot, previous=previous, posture=posture, t=t):
            d.qpos[ad[cols]] = x
            mujoco.mj_forward(m, d)
            gaps = np.array([mujoco.mj_geomDistance(m, d, g, h, 0.03, None) for g, h in pairs])
            return np.r_[
                (d.xpos[wb] + d.xmat[wb].reshape(3, 3) @ rimlocal - target - rot @ rimlocal) * 2000,
                R.from_matrix(d.xmat[wb].reshape(3, 3) @ rot.T).as_rotvec() * 200,
                np.minimum(gaps - 0.0002, 0) * 4000,
                (x - posture) * 0.4,
                (x - previous) * 1.5,
                x[5] * 0.5,
            ]

        initial = (1 - weight) * previous + weight * 0.2 * anchor + weight * 0.8 * previous
        fit = least_squares(
            fun,
            np.clip(initial, lim[:, 0] + 1e-8, lim[:, 1] - 1e-8),
            bounds=(lim[:, 0], lim[:, 1]),
            max_nfev=90,
            diff_step=1e-4,
            ftol=1e-7,
        )
        q[k, cols] = fit.x
        previous = fit.x
        res = fun(fit.x)
        rows.append(
            {
                "t": t,
                "position_error_m": float(np.linalg.norm(res[:3]) / 2000),
                "rotation_error_rad": float(np.linalg.norm(res[3:6]) / 200),
                "cost": fit.cost,
            }
        )
        targets.append({"t": t, "p": target.tolist(), "R": rot.tolist()})
        if k % 50 == 0:
            print(rows[-1], flush=True)
    q[first, cols] = s.joint_pos[first, cols]
    q[last, cols] = s.joint_pos[last, cols]
    refresh_fk(replace(s, joint_pos=q), robot).save(a.out / "motion.npz")
    (a.out / "motion_order.json").write_bytes(Path(info["order"]).read_bytes())
    (a.out / "report.json").write_text(json.dumps(rows, indent=2))
    (a.out / "targets.json").write_text(json.dumps(targets))
    (a.out / "contact_plan.json").write_text(
        json.dumps(
            {
                "open_before_entry": [116, 117],
                "hover": 117.0,
                "palm_seated": 117.7,
                "close": [117.7, 118.7],
                "pull": [118.7, 120.4],
                "open": [120.0, 120.85],
                "vertical_unhook": [120.4, 120.85],
                "outward_after_open": [120.85, 121.2],
                "return_by": 122.8,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
