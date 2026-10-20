"""Rebuild the task-specific right-knee target after ground_stage prepare.

Run ScaleTrack ``retarget3 ground prepare`` with knee_case.yaml and the v3
output directory first. Then run this script before the partial ground solve.
The bias compensates G1 leg proportions; it is not a change to human motion.
"""

from pathlib import Path
from robosimstudio.humanoid_real2sim.paths import example_dir

import numpy as np


ROOT = example_dir(__file__)
TARGETS = ROOT / "retarget/right_knee_ground_v3/intermediate/ground_targets.npz"


def main() -> None:
    with np.load(TARGETS, allow_pickle=False) as file:
        data = {name: file[name] for name in file.files}
    data["knee_xy"] = data["human"][:, 3:5, :2].copy()
    data["knee_xy"][:, 1] += data["gate"][:, 1, None] * np.array([0.10, 0.30])
    np.savez_compressed(TARGETS, **data)
    print(TARGETS)


if __name__ == "__main__":
    main()
