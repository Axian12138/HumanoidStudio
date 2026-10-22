"""Create a rejected-source-fit branch for isolated support diagnostics only."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    import trimesh
    from scipy.spatial import cKDTree
    from scipy.spatial.transform import Rotation

    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--fit", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    assets = args.run / "scene/10_flower_bouquet/registered_assets_paper_wrapper_v1"
    native = args.run / "review/scene/10_native31_inventory"
    fit = json.loads(args.fit.read_text())
    if fit.get("corolla_center_offsets") or fit.get("corolla_scales"):
        raise ValueError("Corolla centre/scale diagnostics require an explicit stem-connection export implementation")
    x = np.array(fit["parameters_length_sx_sy_bendxy_initial_pose_release_pose"])
    branch_tree = None
    secondary_trees = {}
    if fit.get("branch2_diagnostic"):
        branch_vertices = []
        for part in sorted(native.glob("visual_part_*.obj")):
            if 29 <= int(part.name.split("_")[2]) <= 42:
                branch_vertices.append(trimesh.load(part, force="mesh", process=False).vertices)
        branch_tree = cKDTree(np.vstack(branch_vertices))
    for branch_name in fit.get("secondary_branch_offsets", {}):
        branch = int(branch_name)
        parts = [
            trimesh.load(part, force="mesh", process=False).vertices
            for part in sorted(native.glob("visual_part_*.obj"))
            if 1 + 14 * branch <= int(part.name.split("_")[2]) < 15 + 14 * branch
        ]
        secondary_trees[branch] = cKDTree(np.vstack(parts))
    dest, state = args.out / "assets", args.out / "rest_initial_state"
    dest.mkdir()
    state.mkdir()

    def deform(vertices, branch_mask=None, secondary_masks=None):
        v = np.array(vertices, copy=True)
        fraction = np.maximum(0, (v[:, 2] - 0.360078) / (0.68290586 - 0.360078))
        v[:, :2] *= np.where((v[:, 2] > 0.360078)[:, None], x[1:3], 1)
        v[:, :2] += np.clip(fraction * 2, 0, 1)[:, None] * x[3:5]
        v[:, 2] = np.where(v[:, 2] < 0.360078, v[:, 2] * x[0] / 0.360078, x[0] + fraction * (0.68290586 - x[0]))
        if branch_mask is not None:
            v[branch_mask] += np.clip(fraction[branch_mask], 0, 1)[:, None] * x[17:20]
        for branch, mask in (secondary_masks or {}).items():
            delta = np.array(fit["secondary_branch_offsets"][str(branch)])
            v[mask] += np.clip(fraction[mask], 0, 1)[:, None] * delta
        return v

    corolla_transforms = {}
    for branch_name, rotvec in fit.get("corolla_rotvecs", {}).items():
        branch = int(branch_name)
        parts = [
            part
            for part in sorted(native.glob("visual_part_*.obj"))
            if 5 + 14 * branch <= int(part.name.split("_")[2]) <= 14 + 14 * branch
        ]
        flower_vertices = np.vstack([trimesh.load(part, force="mesh", process=False).vertices for part in parts])
        center_part = next(part for part in parts if int(part.name.split("_")[2]) == 5 + 14 * branch)
        center = trimesh.load(center_part, force="mesh", process=False).vertices.mean(axis=0, keepdims=True)
        pivot = deform(
            center,
            np.array([branch == 2]) if branch_tree is not None else None,
            {key: np.array([branch == key]) for key in secondary_trees},
        )[0]
        corolla_transforms[branch] = (cKDTree(flower_vertices), pivot, Rotation.from_rotvec(rotvec))

    # Preserve OBJ material directives and face indices exactly.
    original = (assets / "flower_bouquet.obj").read_text().splitlines()
    coords = np.array([[float(v) for v in line.split()[1:4]] for line in original if line.startswith("v ")])
    transformed_coords = deform(
        coords,
        None if branch_tree is None else branch_tree.query(coords)[0] < 1e-6,
        {branch: tree.query(coords)[0] < 1e-6 for branch, tree in secondary_trees.items()},
    )
    for tree, pivot, rotation in corolla_transforms.values():
        mask = tree.query(coords)[0] < 1e-6
        transformed_coords[mask] = rotation.apply(transformed_coords[mask] - pivot) + pivot
    transformed = iter(transformed_coords)
    rewritten = []
    for line in original:
        if line.startswith("v "):
            rewritten.append("v " + " ".join(f"{v:.10g}" for v in next(transformed)))
        else:
            rewritten.append(line)
    (dest / "flower_bouquet.obj").write_text("\n".join(rewritten) + "\n")
    collision_sources = [assets / "flower_bouquet_hull0.obj", *sorted(native.glob("collision_part_*.obj"))[1:]]
    for index, path in enumerate(collision_sources):
        mesh = trimesh.load(path, force="mesh", process=False)
        mask = np.full(len(mesh.vertices), index in (5, 6)) if branch_tree is not None else None
        mesh.vertices = deform(
            mesh.vertices,
            mask,
            {
                branch: np.full(len(mesh.vertices), index in (1 + 2 * branch, 2 + 2 * branch))
                for branch in secondary_trees
            },
        )
        for branch, (_, pivot, rotation) in corolla_transforms.items():
            if index == 2 + 2 * branch:
                mesh.vertices = rotation.apply(mesh.vertices - pivot) + pivot
        # Piecewise deformation may break original convexity; export its explicit convex envelope.
        trimesh.convex.convex_hull(mesh.vertices).export(dest / f"flower_bouquet_hull{index}.obj")
    for path in assets.iterdir():
        if path.name.startswith("flower_bouquet") and path.suffix == ".obj":
            continue
        (dest / path.name).symlink_to(path.resolve())
    tracks = args.run / "retarget/hoi/10_flower_bouquet/tracks_v8"
    for name in ["flower_bouquet", "small_wicker_basket"]:
        track = np.load(tracks / f"{name}.npz")
        position, quat = track["position_w"][850], track["quaternion_xyzw"][850]
        if name == "flower_bouquet":
            position, quat = x[11:14], Rotation.from_rotvec(x[14:17]).as_quat()
        np.savez_compressed(
            state / f"{name}.npz",
            position_w=np.asarray(position)[None],
            quaternion_xyzw=np.asarray(quat)[None],
            fps=np.array(50),
            source_time_s=np.array([125.0]),
        )
    report = {
        "status": "diagnostic only; source whole-shape fit remains failed, not a deployable candidate",
        "fit": str(args.fit.resolve()),
        "fit_sha256": hashlib.sha256(args.fit.read_bytes()).hexdigest(),
        "shape": "Same source-fit deformation for visual vertices and 30 native foliage/stem hulls; paper is existing frustum with fitted length. No hidden support geometry added.",
        "collision_hulls": len(collision_sources),
        "branch2_diagnostic": branch_tree is not None,
        "secondary_branch_offsets": fit.get("secondary_branch_offsets", {}),
        "limitations": "The tall green sprig is absent from the partial-envelope fit. Branch2 shift follows source-fit visual parts29-42 and native collision parts5-6. The piecewise upper scaling is a diagnostic representation, not a validated connected botanical mesh.",
        "mass": "Existing physical_properties.json and friction .6 unchanged; uniform-solid visual COM is recomputed and remains an unmeasured approximation.",
        "state": "One-frame source125 rest initial conditions only; not a motion trajectory, never integrated into HOI or viewer.",
    }
    (args.out / "provenance.json").write_text(json.dumps(report, indent=2))
    print(args.out)


if __name__ == "__main__":
    main()
