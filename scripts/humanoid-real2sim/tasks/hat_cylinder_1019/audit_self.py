"""Compare nonadjacent right-hand/right-arm self distances with the unchanged baseline."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    import mujoco
    from fit import MODEL

    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--version", default="v5")
    p.add_argument("--source-start-s", type=float, default=35.0)
    a = p.parse_args()
    r = a.run
    m = mujoco.MjModel.from_xml_path(str(MODEL))
    m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    d = mujoco.MjData(m)
    names = json.loads((r / "retarget/baseline/motion_order.json").read_text())["robot_joint_names"]
    addr = [m.joint(n).qposadr[0] for n in names]
    geoms = [g for g in range(m.ngeom) if m.geom_contype[g] and m.geom_bodyid[g] > 0]
    pairs = []

    def ancestors(b):
        ans = {b}
        for _ in range(2):
            b = int(m.body_parentid[b])
            ans.add(b)
        return ans

    for g in geoms:
        bn = m.body(m.geom_bodyid[g]).name
        if not (
            bn.startswith("right_hand_")
            or bn in ["right_elbow_link", "right_wrist_roll_link", "right_wrist_pitch_link", "right_wrist_yaw_link"]
        ):
            continue
        for h in geoms:
            if h == g:
                continue
            b, c = map(int, [m.geom_bodyid[g], m.geom_bodyid[h]])
            if b in ancestors(c) or c in ancestors(b):
                continue
            pair = tuple(sorted((g, h)))
            if pair not in pairs:
                pairs.append(pair)
    arrays = []
    for version in ["baseline", a.version]:
        z = np.load(r / "retarget" / version / "motion.npz")
        rows = []
        frame_count = len(z["joint_pos"])
        for tick in range(2 * frame_count - 1):
            k = tick // 2
            j = min(k + 1, frame_count - 1)
            u = (tick % 2) * 0.5
            d.qpos[addr] = (1 - u) * z["joint_pos"][k] + u * z["joint_pos"][j]
            d.qpos[:3] = (1 - u) * z["body_pos_w"][k, 0] + u * z["body_pos_w"][j, 0]
            qt = (1 - u) * z["body_quat_w"][k, 0] + u * z["body_quat_w"][j, 0]
            d.qpos[3:7] = qt / np.linalg.norm(qt)
            mujoco.mj_forward(m, d)
            rows.append([mujoco.mj_geomDistance(m, d, g, h, 0.05, None) for g, h in pairs])
        arrays.append(np.array(rows))
    old, new = arrays
    bad = np.argwhere((new < -0.001) & (new < old - 0.001))
    report = {
        "method": "100Hz source collision geometries; exclude same and up to two ancestor links; compare old/new signed distances",
        "pairs": len(pairs),
        "worsened_over_1mm_samples": len(bad),
        "worst_candidate_distance_m": float(new.min()),
        "records": [
            {
                "source_s": a.source_start_s + float(i) / 100,
                "bodies": [m.body(m.geom_bodyid[g]).name for g in pairs[j]],
                "old_m": float(old[i, j]),
                "new_m": float(new[i, j]),
            }
            for i, j in bad
        ],
    }
    (r / "review" / ("self_" + a.version + ".json")).write_text(json.dumps(report, indent=2))
    print({k: v for k, v in report.items() if k != "records"})


if __name__ == "__main__":
    main()
