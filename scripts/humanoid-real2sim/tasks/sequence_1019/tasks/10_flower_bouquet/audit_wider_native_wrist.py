"""Independent geometry and posture evidence for the wider basket pull candidate."""


def main():
    import argparse
    import json
    from pathlib import Path

    import mujoco
    import numpy as np

    from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields

    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    r = parser.parse_args().run
    f = r / "retarget/both_v5"
    s = MotionFields.load(f / "motion.npz", f / "motion_order.json")
    m = mujoco.MjModel.from_xml_path(
        "/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof_dex3/g1_29dof_dex3_offsethand.xml"
    )
    d = mujoco.MjData(m)
    a = [m.joint(n).qposadr[0] for n in s.joint_names]
    out = {}
    for side in ["left", "right"]:
        gs = []
        for part in ["elbow_link", "wrist_pitch_link"]:
            b = m.body(side + "_" + part).id
            gs.append(
                next(
                    i for i in range(m.ngeom) if m.geom_bodyid[i] == b and m.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH
                )
            )
        gaps = []
        for t in np.arange(110.8, 126.0001, 0.01):
            k = (t - 108) * 50
            i = min(int(k), len(s.joint_pos) - 2)
            u = k - i
            d.qpos[a] = (1 - u) * s.joint_pos[i] + u * s.joint_pos[i + 1]
            mujoco.mj_forward(m, d)
            gaps.append(mujoco.mj_geomDistance(m, d, *gs, 0.1, None))
        out[side] = {
            "native_mesh_convex_gap_min_m": float(min(gaps)),
            "negative_samples": int((np.array(gaps) < 0).sum()),
            "source_at_min": 110.8 + int(np.argmin(gaps)) * 0.01,
        }
    print(out)
    (r / "review/native_adjacent_wrist_v5.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
