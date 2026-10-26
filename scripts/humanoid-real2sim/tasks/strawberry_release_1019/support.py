"""Project post-release estimated berry settling onto the basket support plane."""


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v8")
    parser.add_argument("--run")
    args = parser.parse_args()
    import json
    from pathlib import Path

    import numpy as np
    import trimesh
    from scipy.spatial.transform import Rotation

    r = Path(args.run) if args.run else Path(Path("/tmp/berry_release_run").read_text())
    p = r / "retarget" / args.version
    z = dict(np.load(p / "table_strawberry.npz"))
    b = np.load(p / "lined_rect_basket.npz")
    br = Rotation.from_quat(b["quaternion_xyzw"]).as_matrix()
    rot = Rotation.from_quat(z["quaternion_xyzw"]).as_matrix()
    verts = np.asarray(
        trimesh.load_mesh(
            "/home/galbot/workspace/ScaleTrack/data/1012test/hoi_stage2/scene/assets/object_strawberry_existing_0_0.obj"
        ).vertices
    )
    lifts = []
    for k in range(351, 376):
        local = (verts @ rot[k].T + z["position_w"][k] - b["position_w"][k]) @ br[k]
        lift = max(0, 0.0084 - local[:, 2].min())
        z["position_w"][k] += br[k][:, 2] * lift
        lifts.append(float(lift))
    np.savez_compressed(p / "table_strawberry.npz", **z)
    (p / "support_projection.json").write_text(
        json.dumps(
            {
                "postrelease_max_support_lift_m": max(lifts),
                "lift_m": lifts,
                "scope": "kinematic support projection; not a fresh physical settling rollout",
            },
            indent=2,
        )
    )
    print(max(lifts))


if __name__ == "__main__":
    main()
