"""Export the task-local 69–76 s Motive skeleton for Stage 3.5."""

from pathlib import Path

import numpy as np

from robosimstudio.humanoid_real2sim.paths import example_dir


HERE = example_dir(__file__)
GLOBAL = HERE.parents[1] / "retarget/ground_support_27_end/sidecars/human_motion_world.npz"


def main(output: Path) -> None:
    with np.load(HERE / "hoi_v3_nonprehensile/human_69_76.npz", allow_pickle=False) as task, np.load(GLOBAL, allow_pickle=False) as source:
        local = np.asarray(task["keypoints_w"], dtype=np.float32)
        aligned = np.asarray(source["positions_w"][2100:2450], dtype=np.float32)
        if local.shape != (350, 20, 3) or not np.allclose(local, aligned, atol=1e-6):
            raise ValueError("task human.npz does not match 69–76 s of continuous Motive trajectory")
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output, positions_w=local, joint_names=source["joint_names"],
            edges=source["edges"], fps=np.asarray(50.0),
            source_range_s=np.asarray([69.0, 76.0]),
        )
    print(f"{output}: 350 frames, source 69.00–75.98 s")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    main(parser.parse_args().output)
