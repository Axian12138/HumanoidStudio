"""Reproduce a task-specific flower HOI refinement step; see the run ledger for inputs."""


def main():

    import argparse
    import json
    from dataclasses import replace
    from pathlib import Path

    import numpy as np
    from scipy.ndimage import gaussian_filter1d

    from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields, refresh_fk

    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    r = parser.parse_args().run
    order = r / "retarget/web_hook_v3/motion_order.json"
    s = MotionFields.load(r / "retarget/web_hook_v3/motion.npz", order)
    left = MotionFields.load(r / "retarget/hook_contact_v5/motion.npz", order)
    q = s.joint_pos.copy()
    cols = [
        i
        for i, n in enumerate(s.joint_names)
        if n.startswith("left_") and any(x in n for x in ["shoulder", "elbow", "wrist", "hand"])
    ]
    q[:, cols] = left.joint_pos[:, cols]
    design = json.load(open(r / "review/grip_design.json"))[0]
    fcols = [s.joint_names.index("right_hand_" + n + "_joint") for n in design["names"]]
    end = round((124.0 - 108) * 50)
    start = 750
    target = s.joint_pos[end, fcols]
    grip = np.array(design["hand"])
    for k in range(start, end + 1):
        u = np.clip((108 + k / 50 - 123.0) / 1.0, 0, 1)
        w = u**3 * (10 - 15 * u + 6 * u * u)
        q[k, fcols] = (1 - w) * grip + w * target
    lf = [i for i, n in enumerate(s.joint_names) if n.startswith("left_hand_")]
    sm = gaussian_filter1d(q[:, lf], 1.5, axis=0)
    q[400:440, lf] = sm[400:440]
    out = r / "retarget/merged_v6"
    out.mkdir()
    robot = Path(
        "/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof_dex3/g1_29dof_dex3_offsethand.xml"
    )
    refresh_fk(replace(s, joint_pos=q), robot).save(out / "motion.npz")
    (out / "motion_order.json").write_bytes(order.read_bytes())


if __name__ == "__main__":
    main()
