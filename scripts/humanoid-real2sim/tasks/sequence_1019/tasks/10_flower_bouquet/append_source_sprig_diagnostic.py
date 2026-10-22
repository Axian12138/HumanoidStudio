"""Add the explicitly estimated green sprig to an isolated diagnostic asset."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    import numpy as np
    import trimesh
    from scipy.spatial.transform import Rotation

    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--base-assets", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--initial-world-translation", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    reference = args.run / "review/scene/10_multiview_flower_envelope_v6_branch/report.json"
    sprig_path = args.run / "review/scene/10_green_sprig_geometry_v2/estimated_sprig.obj"
    x0 = np.array(json.loads(reference.read_text())["parameters_length_sx_sy_bendxy_initial_pose_release_pose"])
    x = np.array(json.loads(args.fit.read_text())["parameters_length_sx_sy_bendxy_initial_pose_release_pose"])
    sprig = trimesh.load(sprig_path, force="mesh", process=False)
    world = Rotation.from_rotvec(x0[8:11]).apply(sprig.vertices) + x0[5:8]
    world += np.asarray(args.initial_world_translation)
    sprig.vertices = Rotation.from_rotvec(x[8:11]).inv().apply(world - x[5:8])
    for path in args.base_assets.iterdir():
        if path.name not in ("flower_bouquet.obj", "flower_bouquet.mtl"):
            (args.out / path.name).symlink_to(path.resolve())
    original = (args.base_assets / "flower_bouquet.obj").read_text()
    vertex_count = sum(line.startswith("v ") for line in original.splitlines())
    extra = ["", "g estimated_green_sprig", "usemtl GreenSprigEstimate"]
    extra.extend("v " + " ".join(f"{v:.10g}" for v in row) for row in sprig.vertices)
    extra.extend("f " + " ".join(str(int(v) + vertex_count + 1) for v in row) for row in sprig.faces)
    (args.out / "flower_bouquet.obj").write_text(original + "\n" + "\n".join(extra) + "\n")
    material = (args.base_assets / "flower_bouquet.mtl").read_text()
    material += "\nnewmtl GreenSprigEstimate\nKd 0.28 0.40 0.27\nKa 0 0 0\nKs 0 0 0\n"
    (args.out / "flower_bouquet.mtl").write_text(material)
    first = 1 + max(int(p.stem.split("hull")[1]) for p in args.base_assets.glob("flower_bouquet_hull*.obj"))
    components = sprig.split(only_watertight=False)
    for index, component in enumerate(components, first):
        trimesh.convex.convex_hull(component.vertices).export(args.out / f"flower_bouquet_hull{index}.obj")
    report = {
        "status": "Diagnostic only; not accepted source geometry or physics",
        "inputs": {
            str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [reference, sprig_path, args.fit, args.base_assets / "flower_bouquet.obj"]
        },
        "scope": "Preserves estimated initial-world sprig geometry; converts it into the candidate bouquet frame.",
        "added_colliders": len(components),
        "initial_world_translation_m": args.initial_world_translation,
        "representation": "One estimated stem and ten thin ellipsoid leaves, each with its own convex collision hull.",
        "limitations": [
            "Stem length, leaf dimensions, and thickness are estimates, not measured botanical geometry.",
            "Rigid leaf approximation does not reproduce actual flexure or establish release branch identity.",
            "Total authored mass remains unchanged; uniform visual-solid COM is recomputed, not a measured mass distribution.",
            "No hidden support plane, friction change, or COM override.",
        ],
    }
    (args.out / "sprig_provenance.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
