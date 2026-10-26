"""Reproduce a task-specific flower HOI refinement step; see the run ledger for inputs."""


def main():

    import argparse
    from dataclasses import replace
    from pathlib import Path

    from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields, refresh_fk

    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    r = parser.parse_args().run
    f = r / "retarget/web_hook_v12"
    s = MotionFields.load(f / "motion.npz", f / "motion_order.json")
    q = s.joint_pos.copy()
    cols = [
        i
        for i, n in enumerate(s.joint_names)
        if n.startswith("left_") and any(v in n for v in ["shoulder", "elbow", "wrist"])
    ]
    q[496, cols] = (q[495, cols] + q[497, cols]) / 2
    out = r / "retarget/web_hook_v13"
    out.mkdir()
    robot = Path(
        "/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof_dex3/g1_29dof_dex3_offsethand.xml"
    )
    refresh_fk(replace(s, joint_pos=q), robot).save(out / "motion.npz")
    (out / "motion_order.json").write_bytes((f / "motion_order.json").read_bytes())


if __name__ == "__main__":
    main()
