"""Search observed-hat poses near the corrected hook without changing hat geometry."""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np


def main():
    import mujoco
    from scipy.spatial import ConvexHull
    from scipy.spatial.transform import Rotation

    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    a = p.parse_args()
    r = a.run
    m = mujoco.MjModel.from_xml_path(str(r / "scene/03_04_05/hat_rack_physics_hook_registered.xml"))
    fit = json.loads((r / "review/scene/03_04_05_hat_multiview_landmarks_fit.json").read_text())["variants"]["rigid"]
    R = Rotation.from_quat(fit["pose_xyzw"][3:])
    g = m.geom("rack_55")
    hook = m.geom_pos[g.id].copy()
    T = np.array(
        next(
            x
            for x in json.loads((r / "review/scene/03_04_05_shared_room_validation/projections.json").read_text())[
                "records"
            ]
            if x["source_s"] == 48
        )["T_world_camera"]
    )
    K = np.array(json.loads((r / "scene/10_flower_bouquet/head_marker_diagnostic_v4_header.json").read_text())["K"])
    obs = np.array(
        json.loads((r / "review/scene/03_04_05_hat_source_pose_comparison.json").read_text())["source_observations_px"]
    )
    v = np.array(
        [
            list(map(float, l.split()[1:]))
            for l in (r / "scene/03_04_05/black_ribbon_hat.obj").read_text().splitlines()
            if l.startswith("v ")
        ]
    )
    records = []

    def score(q):
        world = Rotation.from_quat(q[3:][[1, 2, 3, 0]]).apply(v) + q[:3]
        pc = (world - T[:3, 3]) @ T[:3, :3]
        uv = pc @ K.T
        uv = uv[:, :2] / uv[:, 2:]
        poly = uv[ConvexHull(uv).vertices]
        delta = np.roll(poly, -1, axis=0) - poly
        u = np.clip(((obs[:, None] - poly) * delta).sum(axis=2) / (delta * delta).sum(axis=1), 0, 1)
        err = np.linalg.norm(obs[:, None] - (poly[None] + u[:, :, None] * delta), axis=2).min(axis=1)
        return {"median_px": float(np.median(err)), "max_px": float(err.max())}

    for xyz in itertools.product([0.025, 0.04, 0.055], [0.05, 0.065, 0.08], [0.015, 0.035, 0.055]):
        d = mujoco.MjData(m)
        d.qpos[:3] = hook - R.apply(xyz)
        d.qpos[3:7] = R.as_quat()[[3, 0, 1, 2]]
        mujoco.mj_forward(m, d)
        initial = d.qpos.copy()
        pen = float(min([c.dist for c in d.contact] or [0]))
        samples = []
        for k in range(3000):
            mujoco.mj_step(m, d)
            if k % 50 == 0:
                samples.append(d.qpos.copy())
        samples = np.array(samples)
        rot = Rotation.from_quat(samples[-10:, 3:7][:, [1, 2, 3, 0]])
        record = {
            "hook_local": list(xyz),
            "initial_qpos": initial.tolist(),
            "initial_penetration_m": pen,
            "initial_projection": score(initial),
            "final_qpos": d.qpos.tolist(),
            "final_projection": score(d.qpos),
            "tail1s_max_position_m": float(np.linalg.norm(samples[-10:, :3] - samples[-10, :3], axis=1).max()),
            "tail1s_max_rotation_rad": float((rot * rot[0].inv()).magnitude().max()),
            "root_drop_m": float(d.qpos[2] - initial[2]),
            "final_contacts": d.ncon,
        }
        records.append(record)
        print({k: v for k, v in record.items() if k not in ["initial_qpos", "final_qpos"]}, flush=True)
    (r / "review/scene/03_04_05_source_hook_support_family.json").write_text(
        json.dumps(
            {"status": "support_family_diagnostic_not_acceptance", "hat_shape_unchanged": True, "records": records},
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
