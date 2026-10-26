"""Use a boundary-velocity-matched joint-space withdrawal after fingers are open."""

import argparse
import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np

from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields, refresh_fk


def main():
    from fit import ARM, MODEL
    from scipy.interpolate import BPoly

    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--input-version", default="v6")
    p.add_argument("--output-version", default="v7")
    a = p.parse_args()
    r = a.run
    src = r / "retarget" / a.input_version
    out = r / "retarget" / a.output_version
    shutil.copytree(src, out)
    m = MotionFields.load(src / "motion.npz", src / "motion_order.json")
    base = MotionFields.load(r / "retarget/baseline/motion.npz", src / "motion_order.json")
    q = m.joint_pos.astype(float).copy()
    cols = [m.joint_names.index("right_" + x + "_joint") for x in ARM]
    start = 670
    end = 750
    times = np.arange(len(q)) / 50 + 35
    f = BPoly.from_derivatives(
        [times[start], times[end]],
        [
            [q[start, cols], (q[start, cols] - q[start - 1, cols]) * 50, np.zeros(7)],
            [base.joint_pos[end, cols], base.joint_vel[end, cols], np.zeros(7)],
        ],
    )
    q[start:, cols] = f(times[start:])
    res = refresh_fk(replace(m, joint_pos=q), MODEL)
    res.save(out / "motion.npz")
    report = json.loads((src / "review.json").read_text())
    report.update(
        status="candidate_pending_updated_withdrawal_audits",
        withdrawal={
            "window_s": [48.4, 50],
            "method": "quintic joint-space boundary position and velocity matching; collision audits required",
            "source": "v6 hold / baseline endpoint",
        },
    )
    (out / "review.json").write_text(json.dumps(report, indent=2))
    print(out)


if __name__ == "__main__":
    main()
