"""Check exact source-fit pose under native MuJoCo gravity; not an RSS gate."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    import mujoco
    from scipy.spatial.transform import Rotation

    p = argparse.ArgumentParser()
    p.add_argument("--fit", type=Path, required=True)
    p.add_argument("--seconds", type=float, default=10)
    p.add_argument("--model", type=Path)
    a = p.parse_args()
    fit = json.loads(a.fit.read_text())
    model_path = str(a.model) if a.model else fit["model"]
    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)
    pose = np.array(fit["variants"]["rigid"]["pose_xyzw"])
    adr = m.joint("black_ribbon_hat_free").qposadr[0]
    d.qpos[adr : adr + 3] = pose[:3]
    d.qpos[adr + 3 : adr + 7] = pose[3:][[3, 0, 1, 2]]
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    hat = m.body("black_ribbon_hat").id
    rack = m.body("coat_rack_support").id
    q = []
    times = []
    contacts = []
    for k in range(round(a.seconds / m.opt.timestep) + 1):
        if k % 10 == 0:
            q.append(d.qpos[adr : adr + 7].copy())
            times.append(d.time)
            contacts.append(
                sum({int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])} == {hat, rack} for c in d.contact)
            )
        if k < round(a.seconds / m.opt.timestep):
            mujoco.mj_step(m, d)
    q = np.array(q)
    times = np.array(times)
    rots = Rotation.from_quat(q[:, [4, 5, 6, 3]])
    angle = (rots * rots[0].inv()).magnitude()
    delta = np.linalg.norm(q[:, :3] - q[0, :3], axis=1)
    stem = a.fit.with_name(a.fit.stem + "_native_gravity")
    np.savez_compressed(stem.with_suffix(".npz"), time_s=times, qpos_wxyz=q, rack_contact_count=contacts)
    report = {
        "status": "diagnostic_native_mujoco_not_rss",
        "source_fit": str(a.fit),
        "model": model_path,
        "initial_pose_xyzw": pose.tolist(),
        "duration_s": float(times[-1]),
        "maximum_translation_m": float(delta.max()),
        "maximum_rotation_deg": float(np.degrees(angle).max()),
        "rack_contact_fraction": float(np.mean(np.array(contacts) > 0)),
        "final_rack_contacts": contacts[-1],
        "final_translation_m": float(delta[-1]),
        "final_rotation_deg": float(np.degrees(angle[-1])),
        "tail5_translation_range_m": np.ptp(q[times >= a.seconds - 5, :3], axis=0).tolist(),
    }
    stem.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(report)


if __name__ == "__main__":
    main()
