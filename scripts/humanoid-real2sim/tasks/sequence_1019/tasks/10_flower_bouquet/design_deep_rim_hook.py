"""Place the basket rim at the palm with two curled fingers extending down its inner face."""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def main():
    import mujoco
    from scipy.optimize import least_squares

    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    robot = Path(
        "/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof_dex3/g1_29dof_dex3_offsethand.xml"
    )
    root = ET.parse(robot).getroot()
    root.find("compiler").set("meshdir", str(robot.parent / "meshes"))
    wr = root.find(".//body[@name='left_wrist_yaw_link']")
    body = ET.SubElement(wr, "body", name="rim_probe")
    ET.SubElement(body, "geom", name="rim_wall", type="box", size=".006 .104208 .10", pos="0 -.104208 0")
    m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    d = mujoco.MjData(m)
    m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    ob = m.body("rim_probe").id
    og = m.geom("rim_wall").id
    wb = m.body("left_wrist_yaw_link").id
    names = ["thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1"]
    addr = [m.joint("left_hand_" + n + "_joint").qposadr[0] for n in names]
    parts = ["palm", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1"]
    groups = []
    for part in parts:
        proxy = m.geom("left_hand_" + part + "_collision").id
        b = m.geom_bodyid[proxy]
        ids = [i for i in range(m.ngeom) if m.geom_bodyid[i] == b and m.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH]
        groups.append(list({*ids, proxy}))

    def measure(x):
        m.body_pos[ob] = [x[0], x[1], 0]
        d.qpos[addr] = [0, 0.2, 0.3, -x[2], -x[3], -x[2], -x[3]]
        mujoco.mj_forward(m, d)
        gaps = np.array([min(mujoco.mj_geomDistance(m, d, g, og, 0.3, None) for g in group) for group in groups])
        tips = np.array(
            [
                (d.xpos[m.body("left_hand_" + f + "_finger_tip").id] - d.xpos[wb]) @ d.xmat[wb].reshape(3, 3)
                for f in ["index", "middle"]
            ]
        )
        return gaps, tips

    def fun(x):
        g, t = measure(x)
        return np.r_[
            (g[[0, 4, 6]] - 0.0015) * 500,
            np.minimum(g - 0.0005, 0) * 4000,
            (x[:2] - [0.095, -0.025]) * 30,
            np.maximum(t[:, 1] - x[1] + 0.04, 0) * 300,
            np.maximum(x[0] + 0.006 - t[:, 0], 0) * 500,
        ]

    fits = []
    for base in [0.9, 1.1, 1.3, 1.5]:
        z = least_squares(
            fun,
            [0.1, -0.025, base, 0.4],
            bounds=([0.075, -0.06, 0.7, 0.05], [0.125, -0.005, 1.57, 1.5]),
            max_nfev=300,
            diff_step=1e-4,
        )
        g, t = measure(z.x)
        fits.append(
            {
                "score": float(np.linalg.norm(z.fun)),
                "rim_in_wrist": [*z.x[:2], 0],
                "finger_angles": [0, 0.2, 0.3, -z.x[2], -z.x[3], -z.x[2], -z.x[3]],
                "finger_names": names,
                "gaps": g.tolist(),
                "tip_local": t.tolist(),
                "tip_depth_below_rim_m": (z.x[1] - t[:, 1]).tolist(),
            }
        )
    fits.sort(key=lambda x: x["score"])
    best = fits[0]
    best["opening_sweep"] = [
        {
            "closure": float(u),
            "gaps": measure(
                [
                    best["rim_in_wrist"][0],
                    best["rim_in_wrist"][1],
                    -best["finger_angles"][3] * u,
                    -best["finger_angles"][4] * u,
                ]
            )[0].tolist(),
        }
        for u in np.linspace(0, 1, 21)
    ]
    a.out.write_text(json.dumps(fits, indent=2))
    print(json.dumps(fits[0], indent=2))


if __name__ == "__main__":
    main()
