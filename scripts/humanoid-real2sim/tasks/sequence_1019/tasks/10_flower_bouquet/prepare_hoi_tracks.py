"""Construct explicitly estimated two-object paths for the complete clip 10 action."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    from scipy.signal import savgol_filter
    from scipy.spatial.transform import Rotation, Slerp

    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    folder = a.run / "scene/10_flower_bouquet"
    rigid = np.load(a.run / "inputs/10_flower_bouquet/rigid/wrist_reference.npz")
    t = np.asarray(rigid["recording_time_s"])
    n = len(t)
    basket = json.loads((folder / "basket_registered_candidate.json").read_text())
    flower = json.loads((folder / "flower_registered_candidate.json").read_text())
    bp = np.tile(basket["position_m"], (n, 1))
    bq = np.tile(basket["quaternion_xyzw"], (n, 1))
    fp = np.tile(flower["position_m"], (n, 1))
    fq = np.tile(flower["quaternion_xyzw"], (n, 1))

    def idx(s):
        return int(np.argmin(abs(t - s)))

    def smooth(x):
        x = np.clip(x, 0, 1)
        return x * x * (3 - 2 * x)

    left = savgol_filter(rigid["left_rigid_pos_w"], 21, 3, axis=0)
    right = savgol_filter(rigid["right_rigid_pos_w"], 21, 3, axis=0)
    # The basket stays on the actual support plane while being pulled/pushed.
    pull = np.interp(t, t[idx(119) : idx(120.4) + 1], left[idx(119) : idx(120.4) + 1, 0] - left[idx(119), 0])
    pull *= smooth((t - 118.7) / 0.3)
    push = np.interp(t, t[idx(125) : idx(127.8) + 1], left[idx(125) : idx(127.8) + 1, 0] - left[idx(125), 0])
    push *= smooth((t - 124.7) / 0.3)
    bp[:, 0] += pull + push
    start = idx(113.2)
    end = idx(122.8)
    rhand = Rotation.from_quat(rigid["right_rigid_quat_xyzw"])
    r0 = Rotation.from_quat(flower["quaternion_xyzw"])
    relative = rhand[start].inv() * r0
    # Calibrated relative attachment at the observed wrapped stem; no old wrist camera extrinsic.
    offset = rhand[start].inv().apply(np.array(flower["position_m"]) - right[start])
    carry_r = rhand * relative
    carry_p = right + rhand.apply(np.tile(offset, (n, 1)))
    axis = np.array([0.014, -0.20, 0.979696])
    axis /= np.linalg.norm(axis)
    final_r = Rotation.align_vectors(axis[None], r0.as_matrix()[:, 2][None])[0] * r0
    final_base = bp + np.array([0.03, 0.07, 0.026])
    for k in range(start, n):
        if k <= idx(121):
            fp[k] = carry_p[k]
            fq[k] = carry_r[k].as_quat()
        elif k < end:
            w = smooth((t[k] - 121) / (122.8 - 121))
            fp[k] = (1 - w) * carry_p[k] + w * final_base[k]
            fq[k] = Slerp([0, 1], Rotation.from_quat([carry_r[k].as_quat(), final_r.as_quat()]))([w]).as_quat()[0]
        else:
            fp[k] = final_base[k]
            fq[k] = final_r.as_quat()
    # Blend pickup attachment without altering the initial rest interval.
    for k in range(start, idx(113.6)):
        w = smooth((t[k] - 113.2) / 0.4)
        fp[k] = (1 - w) * np.array(flower["position_m"]) + w * fp[k]
        fq[k] = Slerp([0, 1], Rotation.from_quat([flower["quaternion_xyzw"], fq[k]]))([w]).as_quat()[0]
    for name, pos, quat in [("flower_bouquet", fp, fq), ("small_wicker_basket", bp, bq)]:
        np.savez_compressed(
            a.out / (name + ".npz"),
            position_w=pos.astype(np.float32),
            quaternion_xyzw=quat.astype(np.float32),
            fps=np.array(50),
            source_time_s=t,
        )
    manifest = {
        "objects": [{"name": name, "path": name + ".npz"} for name in ["flower_bouquet", "small_wicker_basket"]],
        "joints": [],
    }
    (a.out / "tracks.json").write_text(json.dumps(manifest, indent=2))
    (a.out / "provenance.json").write_text(
        json.dumps(
            {
                "status": "unreviewed geometric trajectory candidate; not accepted HOI or physical success",
                "source_range_s": [float(t[0]), float(t[-1])],
                "source": "independent hand rigid changes plus supported scene priors",
                "events": {
                    "right_grip": 113.2,
                    "lift": 114.4,
                    "left_pull": [119, 120.4],
                    "flower_place": [121, 122.8],
                    "right_release": 123.4,
                    "left_push": [125, 127.8],
                },
                "unverified": [
                    "body-prior visual gate",
                    "full hand contact and collision",
                    "final flower basket and cabinet-side free rest",
                    "continuous same-camera review",
                ],
                "attachment_offset_right_rigid_m": offset.tolist(),
                "final_flower_axis_world": axis.tolist(),
                "basket_pull_m": float(pull.max()),
                "basket_push_m": float(push.min()),
            },
            indent=2,
        )
    )
    print(a.out)


if __name__ == "__main__":
    main()
