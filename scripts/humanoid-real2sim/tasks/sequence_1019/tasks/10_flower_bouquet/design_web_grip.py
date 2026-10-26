"""Design a task-specific opposed grasp in the thumb/index web, using actual meshes."""

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
    tree = ET.parse(robot)
    root = tree.getroot()
    root.find("compiler").set("meshdir", str(robot.parent / "meshes"))
    vertices = [
        [r * np.cos(t), r * np.sin(t), z - 0.25]
        for z, r in [(0, 0.02), (0.360078, 0.033)]
        for t in np.linspace(0, 2 * np.pi, 64, endpoint=False)
    ]
    ET.SubElement(
        root.find("asset"), "mesh", name="grip_wrapper", vertex=" ".join(str(v) for row in vertices for v in row)
    )
    wrist = root.find(".//body[@name='right_wrist_yaw_link']")
    obj = ET.SubElement(wrist, "body", name="grip_object", pos=".12 .04 0")
    ET.SubElement(obj, "geom", name="grip_shape", type="mesh", mesh="grip_wrapper", contype="1", conaffinity="1")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    ob = model.body("grip_object").id
    og = model.geom("grip_shape").id
    wb = model.body("right_wrist_yaw_link").id
    names = ["thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1"]
    addr = [model.joint("right_hand_" + n + "_joint").qposadr[0] for n in names]
    gs = [
        model.geom("right_hand_" + n + "_collision").id
        for n in ["palm", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1"]
    ]
    pairs = [(gs[i], gs[j]) for i, j in [(1, 3), (1, 4), (1, 5), (1, 6), (2, 3), (2, 4), (2, 5), (2, 6)]]

    def measure(x):
        model.body_pos[ob] = [x[0], x[1], 0]
        data.qpos[addr] = [0, -x[2], -x[3], x[4], x[5], x[4], x[5]]
        mujoco.mj_forward(model, data)
        gaps = np.array([mujoco.mj_geomDistance(model, data, g, og, 0.3, None) for g in gs])
        normals = []
        for i in [2, 4, 6]:
            ft = np.zeros(6)
            mujoco.mj_geomDistance(model, data, gs[i], og, 0.3, ft)
            v = (ft[3:] - data.xpos[ob]) @ data.xmat[wb].reshape(3, 3)
            v = v[:2]
            normals.append(v / max(np.linalg.norm(v), 1e-8))
        selfgap = np.array([mujoco.mj_geomDistance(model, data, g, h, 0.03, None) for g, h in pairs])
        return gaps, np.array(normals), selfgap

    def fun(x):
        gaps, n, sg = measure(x)
        return np.r_[
            (gaps[[2, 4, 6]] - 0.001) * 400,
            np.minimum(gaps - 0.0005, 0) * 2500,
            (n[0] + (n[1] + n[2]) / 2) * 4,
            np.minimum(sg - 0.001, 0) * 1000,
            (x[:2] - [0.11, 0.035]) * 3,
        ]

    results = []
    for center in [[0.11, 0.035], [0.12, 0.055], [0.10, 0.02]]:
        fit = least_squares(
            fun,
            [*center, 0.5, 0.6, 0.55, 0.65],
            bounds=([0.07, -0.01, 0, 0, 0, 0], [0.15, 0.08, 1.0, 1.7, 1.57, 1.74]),
            max_nfev=250,
            diff_step=1e-4,
        )
        gaps, n, sg = measure(fit.x)
        results.append(
            {
                "score": float(np.linalg.norm(fun(fit.x))),
                "center": [*fit.x[:2], 0],
                "hand": [0, -fit.x[2], -fit.x[3], fit.x[4], fit.x[5], fit.x[4], fit.x[5]],
                "names": names,
                "gaps": gaps.tolist(),
                "opposition": (n[0] @ n[1:].T).tolist(),
                "self_min": float(sg.min()),
            }
        )
    results.sort(key=lambda v: v["score"])
    a.out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results[0], indent=2))


if __name__ == "__main__":
    main()
