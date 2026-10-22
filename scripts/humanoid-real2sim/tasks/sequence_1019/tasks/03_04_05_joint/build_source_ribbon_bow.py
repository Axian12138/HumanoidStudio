"""Private source-constrained planar ribbon hypothesis; no acceptance implied."""

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def main():
    import trimesh
    from PIL import Image, ImageDraw
    from scipy.optimize import least_squares

    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-plane", action="store_true")
    ap.add_argument("--cross-phase", action="store_true")
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--output", type=Path)
    ap.add_argument(
        "--release-pose", type=Path, help="JSON with pose_xyzw for an independently documented release-pose control"
    )
    a = ap.parse_args()
    r = a.run.resolve()
    q = r / "inputs/03_04_05/hat_carry_side_source_headers"
    d = r / ("scene/03_04_05/hat_source_ribbon_bow_v2" if a.fit_plane else "scene/03_04_05/hat_source_ribbon_bow_v1")
    if a.cross_phase:
        d = r / "scene/03_04_05/hat_source_ribbon_bow_v3"
    if a.output is not None:
        d = a.output.resolve()
    d.mkdir(exist_ok=False)
    g = {}
    exec((q / "fit_joint_supported_static_carry_source.py").read_text().split("x0=np.r_")[0], g)
    x = np.asarray(json.loads((q / "joint_supported_static_carry_shape_fit.json").read_text())["parameter_vector"])
    y = x[6:12]
    cam = g["ca"]
    rows = json.loads((q / "visible_bow_graph_proposals.json").read_text())["rows"]
    frames = {"1371": "45.7", "1375": "45.833333333333336", "1380": "46.0"}
    keys = [k for k in rows["1375"] if k != "right_tip"]
    src = r / "scene/03_04_05/hat_joint_source_shape_v2_carry_rack/model.xml"
    tree = ET.parse(src)
    asset = tree.getroot().find("asset")
    body = tree.find(".//body[@name='black_ribbon_hat']")
    band = trimesh.load(asset.find("mesh[@name='black_ribbon_hat_visual_part_5']").get("file"), process=False)
    # Anchor must be on the authored band; no invisible free-floating bow attachment.
    v = np.asarray(band.vertices)
    cost = np.zeros(len(v))
    for frame in ["1371", "1380"]:
        cost += np.sum((cam["proj"](v, y, frames[frame]) - rows[frame]["band_entry"]["pixel_xy"]) ** 2, axis=1)
    anchor = v[np.argmin(cost)]
    normal = np.r_[anchor[:2], 0.0]
    normal /= np.linalg.norm(normal)
    basis = np.column_stack((np.cross([0.0, 0.0, 1.0], normal), [0.0, 0.0, 1.0]))
    if a.fit_plane:
        from scipy.spatial.transform import Rotation

        active = [k for k in keys if k != "band_entry"]

        def plane_residual(z):
            bb = Rotation.from_rotvec(z[:3]).as_matrix() @ basis
            pts = anchor + z[3:].reshape(-1, 2) @ bb.T
            return np.concatenate(
                [
                    (cam["proj"](pts, y, frames[f]) - np.array([rows[f][k]["pixel_xy"] for k in active])).ravel()
                    for f in ["1371", "1380"]
                ]
            )

        fits = []
        for rot in [[0, 0, 0], [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]]:
            z = np.r_[rot, np.zeros(len(active) * 2)]
            fits.append(
                least_squares(
                    plane_residual,
                    z,
                    bounds=(
                        np.r_[[-1.5] * 3, [-0.15] * (len(active) * 2)],
                        np.r_[[1.5] * 3, [0.15] * (len(active) * 2)],
                    ),
                    max_nfev=250,
                )
            )
        best = min(fits, key=lambda f: f.cost)
        basis = Rotation.from_rotvec(best.x[:3]).as_matrix() @ basis
        normal = np.cross(basis[:, 0], basis[:, 1])
    coords = {}
    errors = {}
    for key in keys:

        def residual(uv, key=key):
            pt = anchor + basis @ uv
            return np.concatenate(
                [cam["proj"](pt[None], y, frames[f])[0] - rows[f][key]["pixel_xy"] for f in ["1371", "1380"]]
            )

        fit = least_squares(residual, [0, 0], bounds=(-0.15, 0.15))
        coords[key] = anchor + basis @ fit.x
        if key == "band_entry":
            coords[key] = anchor.copy()
        errors[key] = {
            f: float(np.linalg.norm(cam["proj"](coords[key][None], y, frames[f])[0] - rows[f][key]["pixel_xy"]))
            for f in frames
        }
    # Right ends are distinct manual visible endpoints, never the excluded flow tips.
    manual = {
        "right_lower_end": {"1375": [878, 482], "1380": [884, 467]},
        "right_upper_end": {"1375": [862, 480], "1380": [862, 463]},
    }
    for key, obs in manual.items():

        def residual(uv, obs=obs):
            return np.concatenate(
                [cam["proj"]((anchor + basis @ uv)[None], y, frames[f])[0] - point for f, point in obs.items()]
            )

        fit = least_squares(residual, [0, 0], bounds=(-0.15, 0.15))
        coords[key] = anchor + basis @ fit.x
        errors[key] = {
            f: float(np.linalg.norm(cam["proj"](coords[key][None], y, frames[f])[0] - point))
            for f, point in obs.items()
        }
    if a.cross_phase:
        from scipy.spatial.transform import Rotation

        release_pose = np.array(
            json.loads((r / "review/scene/04_ribbon_bow_v2_release_pose.json").read_text())["variants"]["rigid"][
                "pose_xyzw"
            ]
        )
        if a.release_pose is not None:
            release_pose = np.array(json.loads(a.release_pose.read_text())["pose_xyzw"])
        release_camera = np.array(
            json.loads((r / "review/scene/parent_head48_marker_camera_pose_diagnostic.json").read_text())[
                "fitted_T_world_camera"
            ]
        )
        camera_k = np.array(
            json.loads((r / "scene/10_flower_bouquet/head_marker_diagnostic_v4_header.json").read_text())["K"]
        )
        release_obs = {
            "loop_outer_top": [752, 375],
            "loop_outer_left": [751, 378],
            "loop_lower": [755, 383],
            "loop_return": [771, 389],
            "junction": [779, 394],
            "lower_root": [785, 400],
            "lower_mid": [796, 412],
            "lower_tip": [810, 419],
            "right_root": [786, 393],
            "right_mid": [798, 396],
            "right_lower_end": [807, 399],
            "right_upper_end": [802, 397],
        }

        def release_project(point):
            world = Rotation.from_quat(release_pose[3:]).apply(point) + release_pose[:3]
            uv = ((world - release_camera[:3, 3]) @ release_camera[:3, :3]) @ camera_k.T
            return uv[:2] / uv[2]

        for key, observation in release_obs.items():
            initial = coords[key].copy()

            def residual_point(point, key=key, observation=observation):
                terms = []
                if key in manual:
                    terms.extend(cam["proj"](point[None], y, frames[f])[0] - ob for f, ob in manual[key].items())
                else:
                    terms.extend(
                        cam["proj"](point[None], y, frames[f])[0] - rows[f][key]["pixel_xy"] for f in ["1371", "1380"]
                    )
                terms.append(release_project(point) - observation)
                return np.concatenate(terms)

            fit = least_squares(residual_point, initial, bounds=(anchor - 0.15, anchor + 0.15), max_nfev=200)
            coords[key] = fit.x
            for f in frames:
                if key in rows[f]:
                    errors[key][f] = float(
                        np.linalg.norm(cam["proj"](fit.x[None], y, frames[f])[0] - rows[f][key]["pixel_xy"])
                    )
            errors[key]["48"] = float(np.linalg.norm(release_project(fit.x) - observation))
    coords["upper_loop_mid"] = (coords["band_entry"] + coords["junction"]) * 0.5
    paths = {
        "left_loop": (
            [
                "junction",
                "loop_return",
                "loop_lower",
                "loop_outer_left",
                "loop_outer_top",
                "band_entry",
                "upper_loop_mid",
                "junction",
            ],
            [0.0035] * 8,
        ),
        "lower_tail": (["junction", "lower_root", "lower_mid", "lower_tip"], [0.006, 0.008, 0.006, 0.0008]),
        "right_lower_strand": (
            ["junction", "right_root", "right_mid", "right_lower_end"],
            [0.0035, 0.0035, 0.003, 0.001],
        ),
        "right_upper_strand": (["junction", "right_root", "right_upper_end"], [0.0025, 0.0025, 0.0008]),
    }
    # Remove only old two bow parts and their shell cells; crown/brim/band/rack untouched.
    removed = []
    for e in list(asset):
        name = e.get("name", "")
        if name in ["black_ribbon_hat_visual_part_6", "black_ribbon_hat_visual_part_7"] or name.startswith(
            ("hat_shell_part6_", "hat_shell_part7_")
        ):
            removed.append(name)
            asset.remove(e)
    for e in list(body):
        if e.tag == "geom" and e.get("mesh") in removed:
            body.remove(e)
    cells = []
    parts = []
    for label, (nodes, widths) in paths.items():
        segs = []
        for i in range(len(nodes) - 1):
            p0, p1 = [coords[n] for n in nodes[i : i + 2]]
            t = p1 - p0
            if np.linalg.norm(t) < 1e-6:
                continue
            side = np.cross(normal, t)
            side /= np.linalg.norm(side)
            vs = []
            for p, w in [(p0, widths[i]), (p1, widths[i + 1])]:
                for s, z in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
                    vs.append(p + side * s * w / 2 + normal * z * 0.0004)
            cell = trimesh.convex.convex_hull(np.asarray(vs))
            name = f"ribbon_{label}_{i}"
            file = d / f"{name}.obj"
            cell.export(file)
            ET.SubElement(asset, "mesh", name=name, file=str(file))
            ET.SubElement(body, "geom", name=name, type="mesh", mesh=name, group="3", rgba="0 0 0 0", mass="0")
            cells.append(cell)
            segs.append(cell)
        mesh = trimesh.util.concatenate(segs)
        file = d / f"{label}_visual.obj"
        mesh.export(file)
        name = f"ribbon_{label}_visual"
        ET.SubElement(asset, "mesh", name=name, file=str(file))
        ET.SubElement(
            body,
            "geom",
            type="mesh",
            mesh=name,
            contype="0",
            conaffinity="0",
            group="2",
            mass="0",
            rgba="0.88 0.85 0.76 1",
        )
        parts.append(mesh)
    tree.write(d / "model.xml")
    trimesh.util.concatenate(parts).export(d / "bow_visual_union.obj")
    for frame, frame_time in frames.items():
        im = Image.open(q / f"head_mcap_{frame}.jpg")
        draw = ImageDraw.Draw(im)
        for label, (nodes, _widths) in paths.items():
            uv = cam["proj"](np.array([coords[n] for n in nodes]), y, frame_time)
            draw.line(
                [tuple(p) for p in uv],
                fill={
                    "left_loop": "cyan",
                    "lower_tail": "orange",
                    "right_lower_strand": "magenta",
                    "right_upper_strand": "yellow",
                }[label],
                width=2,
            )
        im.save(d / f"projection_{frame}.png")
    report = {
        "status": "unaccepted_loadable_source_hypothesis",
        "source_model": str(src),
        "anchor_policy": "Band mesh vertex minimizing train reprojection of visible band_entry; a candidate attachment, not independently identified seam.",
        "anchor_local_m": anchor.tolist(),
        "plane_normal": normal.tolist(),
        "plane_hypothesis": "Ribbons lie on a shared plane through candidate band attachment; plane orientation fitted when --fit-plane is used. Thickness0.8mm, widths0.8-8mm are explicit estimated representation, not measured source truth.",
        "train": ["1371", "1380", "48"] if a.cross_phase else ["1371", "1380"],
        "cross_phase": a.cross_phase,
        "release_pose_control": str(a.release_pose.resolve()) if a.release_pose else None,
        "release_pose_control_sha256": hashlib.sha256(a.release_pose.read_bytes()).hexdigest()
        if a.release_pose
        else None,
        "cross_phase_limit": "When enabled, independently fitted release pose heldfixed; nodes may depart shared plane. No shape or pose acceptance implied.",
        "holdout": ["1375"],
        "right_endpoint_exception": "Right endpoints manually annotated in1375/1380, neither is a holdout for these two nodes. Original excluded flow endpoints never used.",
        "manual_right_endpoints": manual,
        "node_reprojection_px": errors,
        "local_nodes": {k: v.tolist() for k, v in coords.items()},
        "source_constraints": [
            "Visible left ring opening",
            "Visible broad tapered lower tail",
            "Central overlap",
            "Two open right strands, connection unknown",
        ],
        "unresolved": [
            "Physical seam/junction identity",
            "Nonplanar drape and depth",
            "Crown fold metric depth and static versus grasp deformation",
            "Hidden rack/crown contact",
        ],
        "inertial": "Original explicit inertial and friction unchanged; first comparison isolates geometry.",
        "collision_representation": "Each visual segment equals its closed convex collision cell; no convex hull across left-loop cavity. Segment overlaps remain part of ribbon union.",
        "cell_count": len(cells),
        "all_cells_watertight": all(c.is_watertight for c in cells),
        "source_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
        "model_sha256": hashlib.sha256((d / "model.xml").read_bytes()).hexdigest(),
    }
    (d / "review.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(d),
                "anchor": anchor.tolist(),
                "max_node_error": max(z for v in errors.values() for z in v.values()),
                "cells": len(cells),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
