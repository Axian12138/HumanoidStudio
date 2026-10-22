"""Project all authored hook balls without treating hidden symmetric tips as measured."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    import mujoco
    from PIL import Image, ImageDraw
    from scipy.spatial.transform import Rotation

    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    a = p.parse_args()
    r = a.run
    out = r / "review/scene/03_04_05_rack_support_hypotheses"
    out.mkdir(exist_ok=True)
    records = json.loads((r / "review/scene/03_04_05_shared_room_validation/projections.json").read_text())["records"]
    K = np.array(json.loads((r / "scene/10_flower_bouquet/head_marker_diagnostic_v4_header.json").read_text())["K"])
    raw = json.loads((r / "inputs/03_04_05/rack_empty_hooks/head_raw_pose_audit_header.json").read_text())["rows"]
    E = np.array(
        json.loads((r / "scene/10_flower_bouquet/head_marker_diagnostic_v4_header.json").read_text())[
            "T_head_rigid_camera_optical"
        ]
    )
    C = Rotation.from_euler("z", 90, degrees=True).as_matrix() @ np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    for t in [45, 46]:
        row = raw[str(float(t))]
        T = np.eye(4)
        T[:3, :3] = C @ Rotation.from_quat(row["quaternion_xyzw_raw"]).as_matrix()
        T[:3, 3] = C @ row["position_raw"]
        T[2, 3] += 0.08245383203029633
        records.append({"source_s": t, "T_world_camera": (T @ E).tolist()})
    parts = json.loads((r / "scene/03_04_05/rack_support.json").read_text())["parts"]
    balls = [p for p in parts if "hook_ball" in p["source_prim"]]
    all_records = []
    for version in ["early_registered", "hook_registered"]:
        m = mujoco.MjModel.from_xml_path(str(r / f"scene/03_04_05/hat_rack_physics_{version}.xml"))
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        for t in [44, 45, 46, 47, 48]:
            T = np.array(next(p for p in records if p["source_s"] == t)["T_world_camera"])
            im = Image.open(
                r
                / (
                    f"inputs/03_04_05/rack_empty_hooks/head_{t}_header.jpg"
                    if t in [45, 46]
                    else f"inputs/03_04_05/head_{t}_header.jpg"
                )
            )
            draw = ImageDraw.Draw(im)
            points = []
            for part in balls:
                g = m.geom(part["geom"]).id
                world = d.geom_xpos[g]
                c = (world - T[:3, 3]) @ T[:3, :3]
                uvw = K @ c
                uv = uvw[:2] / uvw[2]
                points.append(
                    {
                        "geom": part["geom"],
                        "source_prim": part["source_prim"],
                        "world_m": world.tolist(),
                        "pixel": uv.tolist(),
                        "camera_depth_m": float(c[2]),
                    }
                )
                if c[2] > 0 and 0 <= uv[0] < im.width and 0 <= uv[1] < im.height:
                    x, y = map(float, uv)
                    color = (255, 80, 40) if part["geom"] == "rack_55" else (0, 230, 255)
                    draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline=color, width=2)
                    draw.text((x + 6, y), part["geom"], fill=color)
            path = out / f"head_{t}_{version}_all_ball_hypotheses.png"
            im.save(path)
            all_records.append(
                {"source_s": t, "variant": version, "T_world_camera": T.tolist(), "points": points, "image": str(path)}
            )
    (out / "report.json").write_text(
        json.dumps(
            {
                "status": "authored_support_hypotheses_not_verified_hidden_tip_positions",
                "selected_tip": "rack_55",
                "selected_tip_basis": "Third lowest hook inferred from original asset symmetry; only other two tips independently compared to source",
                "records": all_records,
                "limitations": [
                    "Image projection does not establish occluded tip contact",
                    "Rod and brim can form multiple support contacts",
                    "All upper hooks retain original asset placement, not independently registered",
                ],
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
