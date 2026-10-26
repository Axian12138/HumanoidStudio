"""Remove the measured 40.6s IK splice; preserve approach and hand timing."""


def main():
    import argparse
    import json
    import shutil
    from dataclasses import replace
    from pathlib import Path

    import numpy as np
    from fit import ARM, MODEL, smooth
    from scipy.ndimage import gaussian_filter1d
    from scipy.spatial.transform import Rotation as R

    from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields, refresh_fk

    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    a = p.parse_args()
    r = a.run
    src = r / "retarget/enclosing_v1"
    out = r / "retarget/enclosing_v2"
    shutil.copytree(src, out)
    m = MotionFields.load(src / "motion.npz", src / "motion_order.json")
    q = m.joint_pos.astype(float).copy()
    cols = [m.joint_names.index("right_" + x + "_joint") for x in ARM]
    tt = 35 + np.arange(len(q)) / 50
    alpha = np.array([smooth(t, 40.15, 40.4) * (1 - smooth(t, 40.85, 41.1)) for t in tt])
    filtered = gaussian_filter1d(q[:, cols], 4, axis=0, mode="nearest")
    q[:, cols] += alpha[:, None] * (filtered - q[:, cols])
    res = refresh_fk(replace(m, joint_pos=q), MODEL)
    res.save(out / "motion.npz")
    design = json.load(open(r / "review/grasp_design_enclosing_v1.json"))
    rel = R.from_quat(design["relative_hat_rotation_xyzw"])
    center = design["selected"]["center"]
    wi = m.body_names.index("right_wrist_yaw_link")
    wr = R.from_quat(res.body_quat_w[:, wi][:, [1, 2, 3, 0]])
    hr = wr * rel
    hp = res.body_pos_w[:, wi] + wr.apply(center) - hr.apply([0, 0, 0.11])
    path = out / "scene_tracks/black_ribbon_hat_free.npz"
    h = np.load(path)
    data = {k: h[k] for k in h.files}
    mask = alpha > 0
    data["position_w"][mask] = hp[mask]
    data["quaternion_xyzw"][mask] = hr.as_quat()[mask]
    np.savez_compressed(path, **data)
    review = json.load(open(src / "review.json"))
    review.update(
        status="candidate_pending_full_geometry_and_semantic_review", parent=str(src), local_smoothing_s=[40.15, 41.1]
    )
    (out / "review.json").write_text(json.dumps(review, indent=2))


if __name__ == "__main__":
    main()
