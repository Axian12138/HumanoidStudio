"""Smooth the elbow correction and clip robot joint bounds before contact revalidation."""


def main():
    import shutil
    from dataclasses import replace
    from pathlib import Path

    import numpy as np
    from scipy.ndimage import gaussian_filter1d

    from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields, refresh_fk

    r = Path(Path("/tmp/wipes_extended_run").read_text())
    src = r / "retarget/v11"
    out = r / "retarget/v13"
    shutil.copytree(src, out)
    b = MotionFields.load(src / "motion.npz", src / "motion_order.json")
    old = MotionFields.load(r / "retarget/v4/motion.npz", src / "motion_order.json")
    q = old.joint_pos.astype(float).copy()
    ids = [
        i
        for i, n in enumerate(b.joint_names)
        if n.startswith("left_") and any(s in n for s in ["shoulder", "elbow", "wrist"])
    ]
    c = gaussian_filter1d((b.joint_pos - old.joint_pos)[:, ids], 2.5, axis=0)
    c[:176] = 0
    c[225:] = 0
    for k in range(176, 183):
        c[k] *= ((k - 175) / 8) ** 2
    for k in range(218, 225):
        c[k] *= ((225 - k) / 8) ** 2
    q[:, ids] += c
    model = Path(
        "/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof_dex3/g1_29dof_dex3_offsethand.xml"
    )
    import mujoco

    m = mujoco.MjModel.from_xml_path(str(model))
    for i in ids:
        q[:, i] = np.clip(q[:, i], *m.joint(b.joint_names[i]).range)
    refresh_fk(replace(b, joint_pos=q), model).save(out / "motion.npz")


if __name__ == "__main__":
    main()
