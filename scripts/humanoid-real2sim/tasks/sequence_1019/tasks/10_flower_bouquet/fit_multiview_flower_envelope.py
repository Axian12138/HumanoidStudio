"""Source-only joint shape/pose diagnostic; never writes assets or tracks."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    import trimesh
    from PIL import Image, ImageDraw
    from scipy.optimize import least_squares
    from scipy.spatial import ConvexHull, cKDTree
    from scipy.spatial.transform import Rotation

    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--support-prior", action="store_true")
    parser.add_argument("--relative-markers", action="store_true")
    parser.add_argument("--balanced-outline", action="store_true")
    parser.add_argument("--wall-clearance", action="store_true")
    parser.add_argument("--initial-native-floor-clearance", action="store_true")
    parser.add_argument("--shared-corolla-tilt", type=float, nargs=3)
    parser.add_argument("--initial-empty-table-regions", action="store_true")
    parser.add_argument("--source-corolla-centers", action="store_true")
    parser.add_argument("--initial-roll-seed-deg", type=float, default=0)
    parser.add_argument("--branch2-diagnostic", action="store_true")
    parser.add_argument("--green-tip-hypothesis", action="store_true")
    parser.add_argument("--secondary-branches", action="store_true")
    parser.add_argument("--seed-fit", type=Path)
    parser.add_argument("--green-wall-envelope", action="store_true")
    parser.add_argument("--initial-detail-view", action="store_true")
    args = parser.parse_args()
    secondary_branches = [4, 6, 8, 11] if args.source_corolla_centers else [4, 6]
    if args.source_corolla_centers and not (args.secondary_branches and args.shared_corolla_tilt):
        parser.error("--source-corolla-centers requires --secondary-branches and --shared-corolla-tilt")
    args.out.mkdir(parents=True, exist_ok=False)
    run = args.run
    scene = run / "scene/10_flower_bouquet"
    cal = json.loads((scene / "head_marker_diagnostic_v4_header.json").read_text())
    heads = json.loads((run / "inputs/10_flower_bouquet/head_raw_pose_audit_header.json").read_text())["rows"]
    release = json.loads((run / "review/scene/10_release_foliage_fit_v1/report.json").read_text())
    outlines = {float(t): points for t, points in release["observations"].items()}
    outlines[111.0] = [
        [686, 72],
        [706, 96],
        [724, 119],
        [765, 133],
        [793, 148],
        [801, 167],
        [797, 196],
        [770, 218],
        [744, 237],
        [716, 241],
        [691, 228],
        [668, 226],
        [639, 220],
        [631, 210],
        [633, 187],
        [648, 167],
        [670, 156],
        [674, 132],
        [678, 101],
    ]
    outlines[112.0] = [
        [590, 163],
        [610, 177],
        [626, 209],
        [665, 237],
        [704, 254],
        [740, 279],
        [759, 313],
        [752, 359],
        [723, 396],
        [688, 424],
        [645, 428],
        [606, 418],
        [565, 388],
        [520, 381],
        [500, 363],
        [497, 336],
        [515, 294],
        [551, 270],
        [567, 240],
        [573, 207],
    ]
    if args.branch2_diagnostic:
        # The separate tall green sprig is not represented by a native white
        # flower stem. Exclude that sprig from this partial-envelope diagnostic.
        outlines[111.0] = outlines[111.0][3:17]
        outlines[112.0] = outlines[112.0][3:17]
    if args.initial_detail_view:
        outlines[113.0] = [
            [502, 312],
            [620, 314],
            [681, 327],
            [733, 332],
            [786, 357],
            [783, 454],
            [740, 475],
            [687, 511],
            [616, 496],
            [569, 493],
            [549, 454],
            [516, 421],
            [466, 403],
            [455, 374],
            [488, 351],
        ]
        # Independent, distant initial view: coarse annotation, not fitted.
        outlines[110.0] = [
            [770, 80],
            [793, 53],
            [830, 43],
            [863, 39],
            [886, 47],
            [896, 63],
            [886, 86],
            [860, 105],
            [814, 116],
            [796, 99],
            [772, 101],
        ]
    family = json.loads((run / "review/scene/10_release_pose_family_v1/report.json").read_text())
    endpoints = dict(zip(family["source_times"], family["observed_pixels_tip_top"], strict=True))
    endpoints.update({111.0: [[912, 129], [798, 158]], 112.0: [[928, 261], [752, 293]]})
    if args.initial_detail_view:
        endpoints.update({110.0: [[982, 49], [890, 66]], 113.0: [[1052, 377], [792, 384]]})
    times = sorted(outlines)
    train = [111.0, 112.0, 123.4, 124.0]
    if args.initial_detail_view:
        train.append(113.0)
    common_residual = {t: np.zeros(2) for t in times}
    if args.relative_markers:
        predicted = json.loads((run / "review/scene/10_release_marker_holdout_v1/projection.json").read_text())
        measured = json.loads((run / "review/scene/10_release_image_delay_v2/report.json").read_text())["rows"]
        for t in times:
            if t < 120:
                continue
            pred = next(row for row in predicted if row["source_s"] == t)
            obs = next(row for row in measured if row["source_s"] == t)
            common_residual[t] = np.mean(np.array(pred["uv"])[:3] - np.array(obs["observed_uv"]), axis=0)
    K = np.array(cal["K"])
    C = Rotation.from_euler("z", 90, degrees=True).as_matrix() @ np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    cameras = {}
    for t in times:
        h = heads[str(t)]
        H = np.eye(4)
        H[:3, :3] = C @ Rotation.from_quat(h["quaternion_xyzw_raw"]).as_matrix()
        H[:3, 3] = C @ h["position_raw"]
        H[2, 3] += 0.08245383203029633
        cameras[t] = H @ cal["T_head_rigid_camera_optical"]
    mesh = trimesh.load(scene / "registered_assets_paper_wrapper_v1/flower_bouquet.obj", force="mesh", process=False)
    v = mesh.vertices[mesh.vertices[:, 2] > 0.360078]
    _, ids = np.unique(np.round(v / 0.008), axis=0, return_index=True)
    vertices = v[ids]
    corolla_groups = []
    if args.shared_corolla_tilt:
        inventory = run / "review/scene/10_native31_inventory"
        for branch in range(12):
            parts = [
                part
                for part in inventory.glob("visual_part_*.obj")
                if 5 + 14 * branch <= int(part.name.split("_")[2]) <= 14 + 14 * branch
            ]
            points = np.vstack([trimesh.load(part, force="mesh", process=False).vertices for part in parts])
            center_part = next(part for part in parts if int(part.name.split("_")[2]) == 5 + 14 * branch)
            center = trimesh.load(center_part, force="mesh", process=False).vertices.mean(axis=0)
            corolla_groups.append((branch, cKDTree(points).query(vertices)[0] < 1e-6, center))
    branch_mask = np.zeros(len(vertices), dtype=bool)
    secondary_masks = []
    if args.branch2_diagnostic:
        branch_vertices = []
        inventory = run / "review/scene/10_native31_inventory"
        for part in inventory.glob("visual_part_*.obj"):
            index = int(part.name.split("_")[2])
            if 29 <= index < 43:
                branch_vertices.append(trimesh.load(part, force="mesh", process=False).vertices)
        branch_mask = cKDTree(np.vstack(branch_vertices)).query(vertices)[0] < 1e-6
    if args.secondary_branches:
        if not args.branch2_diagnostic:
            parser.error("--secondary-branches requires --branch2-diagnostic")
        inventory = run / "review/scene/10_native31_inventory"
        for branch in secondary_branches:
            parts = [
                trimesh.load(part, force="mesh", process=False).vertices
                for part in inventory.glob("visual_part_*.obj")
                if 1 + 14 * branch <= int(part.name.split("_")[2]) < 15 + 14 * branch
            ]
            secondary_masks.append(cKDTree(np.vstack(parts)).query(vertices)[0] < 1e-6)
    room = np.array(json.loads((scene / "shared_level_room_floor_v3.json").read_text())["T_room_correction"])
    side = trimesh.load(scene / "registered_assets_paper_wrapper_v1/cabinet_side.obj", force="mesh", process=False)
    side_local = (side.vertices - room[:3, 3]) @ room[:3, :3]
    side_y = side_local[:, 1].max()
    track = np.load(run / "retarget/hoi/10_flower_bouquet/tracks_v8/flower_bouquet.npz")
    poses = []
    for t in [111.0, 125.0]:
        idx = int(round((t - 108) * 50))
        poses.extend(track["position_w"][idx])
        rotation = Rotation.from_quat(track["quaternion_xyzw"][idx])
        if t == 111.0:
            rotation = rotation * Rotation.from_euler("z", args.initial_roll_seed_deg, degrees=True)
        poses.extend(rotation.as_rotvec())
    start = np.r_[0.360078, 1.0, 1.0, 0.0, 0.0, poses]
    if args.branch2_diagnostic:
        start = np.r_[start, np.zeros(3)]
    if args.secondary_branches:
        start = np.r_[start, np.zeros(3 * len(secondary_branches))]
    if args.seed_fit:
        seed = json.loads(args.seed_fit.read_text())["parameters_length_sx_sy_bendxy_initial_pose_release_pose"]
        start[: len(seed)] = seed
    angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    directions = np.c_[np.cos(angles), np.sin(angles)]

    def clip(poly):
        result = []
        for a, b in zip(poly, np.roll(poly, -1, axis=0), strict=True):
            if a[1] >= 0:
                result.append(a)
            if (a[1] >= 0) != (b[1] >= 0):
                result.append(a + (b - a) * (-a[1]) / (b[1] - a[1]))
        return np.array(result)

    def signature(poly):
        return np.max(poly @ directions.T, axis=0)

    observed = {t: np.array(outlines[t])[ConvexHull(outlines[t]).vertices] for t in times}
    green_tip_world = np.array([-2.8343277639272118, 0.24447642557603338, 0.8390510139966433])
    green_world = None
    if args.green_wall_envelope:
        green_mesh = trimesh.load(
            run / "review/scene/10_green_sprig_geometry_v2/estimated_sprig.obj", force="mesh", process=False
        )
        green_fit = np.array(
            json.loads((run / "review/scene/10_multiview_flower_envelope_v6_branch/report.json").read_text())[
                "parameters_length_sx_sy_bendxy_initial_pose_release_pose"
            ]
        )
        green_world = Rotation.from_rotvec(green_fit[8:11]).apply(green_mesh.vertices) + green_fit[5:8]

    def predict_green_tip(x, t):
        local = Rotation.from_rotvec(x[8:11]).inv().apply(green_tip_world - x[5:8])
        pose = x[5:11] if t < 120 else x[11:17]
        world = Rotation.from_rotvec(pose[3:]).apply(local) + pose[:3]
        cam = cameras[t]
        uvw = ((world - cam[:3, 3]) @ cam[:3, :3]) @ K.T
        return uvw[:2] / uvw[2] - common_residual[t]

    def shaped_vertices(x):
        local = vertices.copy()
        fraction = (local[:, 2] - 0.360078) / (0.68290586 - 0.360078)
        local[:, :2] *= x[1:3]
        local[:, :2] += np.clip(fraction * 2, 0, 1)[:, None] * x[3:5]
        local[:, 2] = x[0] + fraction * (0.68290586 - x[0])
        if args.branch2_diagnostic:
            local[branch_mask] += np.clip(fraction[branch_mask], 0, 1)[:, None] * x[17:20]
        for i, mask in enumerate(secondary_masks):
            local[mask] += np.clip(fraction[mask], 0, 1)[:, None] * x[20 + 3 * i : 23 + 3 * i]
        for branch, mask, original_center in corolla_groups:
            center = original_center.copy()
            frac = (center[2] - 0.360078) / (0.68290586 - 0.360078)
            center[:2] = center[:2] * x[1:3] + np.clip(frac * 2, 0, 1) * x[3:5]
            center[2] = x[0] + frac * (0.68290586 - x[0])
            if branch == 2 and args.branch2_diagnostic:
                center += np.clip(frac, 0, 1) * x[17:20]
            if branch in secondary_branches and args.secondary_branches:
                i = secondary_branches.index(branch)
                center += np.clip(frac, 0, 1) * x[20 + 3 * i : 23 + 3 * i]
            local[mask] = Rotation.from_rotvec(args.shared_corolla_tilt).apply(local[mask] - center) + center
        return local

    def predict(x, t):
        local = shaped_vertices(x)
        local = np.vstack([local, [0, 0, 0], [0, 0, x[0]]])
        p = x[5:11] if t < 120 else x[11:17]
        world = Rotation.from_rotvec(p[3:]).apply(local) + p[:3]
        cam = cameras[t]
        uvw = ((world - cam[:3, 3]) @ cam[:3, :3]) @ K.T
        uv = uvw[:, :2] / uvw[:, 2:]
        uv -= common_residual[t]
        poly = clip(uv[:-2][ConvexHull(uv[:-2]).vertices])
        return poly, uv[-2:]

    def residual(x):
        values = []
        if args.initial_native_floor_clearance:
            # The upper native flower geometry must also clear the measured
            # cabinet top. This is a sampled geometric constraint, not a rest
            # stability objective; the exported full meshes need validation.
            initial_world = Rotation.from_rotvec(x[8:11]).apply(shaped_vertices(x)) + x[5:8]
            values.append(min(0, initial_world[:, 2].min() - 0.661) / 0.001)
        if args.initial_empty_table_regions:
            # Conservative source113 patches visibly contain bare table.
            # These negative observations constrain actual sampled vertices,
            # unlike the whole-bouquet convex envelope.
            world = Rotation.from_rotvec(x[8:11]).apply(shaped_vertices(x)) + x[5:8]
            cam = cameras[113.0]
            q = ((world - cam[:3, 3]) @ cam[:3, :3]) @ K.T
            uv = q[:, :2] / q[:, 2:]
            for left, top, right, bottom in [(490, 420, 535, 478)]:
                inside_depth = np.minimum.reduce(
                    [uv[:, 0] - left, right - uv[:, 0], uv[:, 1] - top, bottom - uv[:, 1]]
                )
                values.extend(np.maximum(0, inside_depth) / 5)
        for t in train:
            poly, ends = predict(x, t)
            # Support-function directions are correlated, not 24 independent labels.
            outline_scale = 12 * (np.sqrt(3) if args.balanced_outline else 1)
            values.extend((signature(poly) - signature(observed[t])) / outline_scale)
            endpoint_sigma = np.array([[5], [8]]) if args.balanced_outline and t < 120 else 12
            if t != 113.0:
                # At113 the right hand occludes the paper lip; its endpoint
                # annotation is shown for context but is not an IK target.
                values.extend(((ends - endpoints[t]) / endpoint_sigma).ravel())
        # Weak representation priors; neither contact nor dynamics is optimized.
        values.extend([(x[0] - 0.360078) / 0.055, (x[1] - 1) / 0.5, (x[2] - 1) / 0.5])
        values.extend(x[3:5] / 0.08)
        if args.branch2_diagnostic:
            values.extend(x[17:20] / 0.15)
        if args.secondary_branches:
            values.extend(x[20 : 20 + 3 * len(secondary_branches)] / 0.08)
        if args.source_corolla_centers:
            # Independently read disk-centre regions, not exact flower-stamen
            # points. 113 is deliberately excluded from this centre fit.
            observations = {
                111.0: {11: ([662, 166], 4), 8: ([711, 157], 4)},
                112.0: {11: ([543, 310], 5), 8: ([624, 291], 5)},
            }
            for t, groups in observations.items():
                for branch, (target, radius) in groups.items():
                    original = next(center for b, _, center in corolla_groups if b == branch)
                    center = original.copy()
                    frac = (center[2] - 0.360078) / (0.68290586 - 0.360078)
                    center[:2] = center[:2] * x[1:3] + np.clip(frac * 2, 0, 1) * x[3:5]
                    center[2] = x[0] + frac * (0.68290586 - x[0])
                    i = secondary_branches.index(branch)
                    center += np.clip(frac, 0, 1) * x[20 + 3 * i : 23 + 3 * i]
                    world = Rotation.from_rotvec(x[8:11]).apply(center) + x[5:8]
                    cam = cameras[t]
                    q = ((world - cam[:3, 3]) @ cam[:3, :3]) @ K.T
                    error = q[:2] / q[2] - target
                    values.extend(np.maximum(0, np.abs(error) - radius) / 2)
        if args.green_tip_hypothesis:
            values.extend((predict_green_tip(x, 123.4) - [995, 35]) / 12)
        if green_world is not None:
            local_green = Rotation.from_rotvec(x[8:11]).inv().apply(green_world - x[5:8])
            release_green = Rotation.from_rotvec(x[14:17]).apply(local_green) + x[11:14]
            room_green = (release_green - room[:3, 3]) @ room[:3, :3]
            values.extend(
                [
                    min(0, room_green[:, 0].min() + 3.04) / 0.001,
                    min(0, room_green[:, 1].min() - side_y) / 0.001,
                ]
            )
        if args.support_prior:
            # Measured cabinet top and basket floor constrain base height,
            # independently of whether an eventual released pose is stable.
            values.extend([(x[7] - 0.68) / 0.005, (x[13] - 0.68) / 0.005])
        if args.wall_clearance:
            world = Rotation.from_rotvec(x[14:17]).apply(shaped_vertices(x)) + x[11:14]
            room_local = (world - room[:3, 3]) @ room[:3, :3]
            values.extend(
                [
                    min(0, room_local[:, 0].min() + 3.04) / 0.001,
                    min(0, room_local[:, 1].min() - side_y) / 0.001,
                ]
            )
            theta = np.arange(32) * 2 * np.pi / 32
            paper = np.vstack(
                [
                    np.c_[0.020 * np.cos(theta), 0.020 * np.sin(theta), np.zeros(32)],
                    np.c_[0.033 * np.cos(theta), 0.033 * np.sin(theta), np.full(32, x[0])],
                ]
            )
            for pose, floor in [(x[5:11], 0.6615), (x[11:17], 0.6735)]:
                paper_world = Rotation.from_rotvec(pose[3:]).apply(paper) + pose[:3]
                values.append((paper_world[:, 2].min() - floor) / 0.001)
        return np.array(values)

    lower = np.r_[0.25, 0.5, 0.5, -0.20, -0.20, np.array(poses) - np.tile([0.25, 0.25, 0.25, 1.2, 1.2, 1.2], 2)]
    upper = np.r_[0.40, 2.0, 2.0, 0.20, 0.20, np.array(poses) + np.tile([0.25, 0.25, 0.25, 1.2, 1.2, 1.2], 2)]
    if args.branch2_diagnostic:
        lower = np.r_[lower, [-0.20, -0.20, -0.08]]
        upper = np.r_[upper, [0.20, 0.20, 0.08]]
    if args.secondary_branches:
        lower = np.r_[lower, np.tile([-0.15, -0.15, -0.08], len(secondary_branches))]
        upper = np.r_[upper, np.tile([0.15, 0.15, 0.08], len(secondary_branches))]
    fit = least_squares(residual, start, bounds=(lower, upper), max_nfev=400, diff_step=1e-4)
    rows = []
    for t in times:
        poly, ends = predict(fit.x, t)
        errors = signature(poly) - signature(observed[t])
        rows.append(
            {
                "source_s": t,
                "partition": "fit" if t in train else "holdout",
                "outline_rms_px": float(np.sqrt(np.mean(errors**2))),
                "outline_max_px": float(np.max(abs(errors))),
                "endpoint_errors_px": np.linalg.norm(ends - endpoints[t], axis=1).tolist(),
            }
        )
        im = Image.open(run / f"inputs/10_flower_bouquet/header_views/{t}.jpg")
        draw = ImageDraw.Draw(im)
        for line, color in [(observed[t], "yellow"), (poly, "cyan")]:
            draw.line([tuple(p) for p in np.vstack([line, line[0]])], fill=color, width=2)
        for p in ends:
            draw.ellipse((p[0] - 4, p[1] - 4, p[0] + 4, p[1] + 4), fill="magenta")
        if args.green_tip_hypothesis:
            p = predict_green_tip(fit.x, t)
            draw.ellipse((p[0] - 6, p[1] - 6, p[0] + 6, p[1] + 6), outline="red", width=2)
        im.save(args.out / f"{t}_projection.png")
    report = {
        "status": "diagnostic only; no asset, track, camera or mass changed",
        "parameters_length_sx_sy_bendxy_initial_pose_release_pose": fit.x.tolist(),
        "rows": rows,
        "observations": outlines,
        "endpoint_observations": endpoints,
        "notes": (
            "Common native shape with a separately bent branch; initial partial envelope excludes the tall green sprig. "
            if args.branch2_diagnostic
            else "Single common native shape; initial full envelope. "
        )
        + "Fixed total length, two static object poses and fixed camera. Manual convex envelopes include empty branch space and do not prove solid support.",
        "optimizer_success": bool(fit.success),
        "optimizer_message": fit.message,
        "support_prior": args.support_prior,
        "relative_markers": args.relative_markers,
        "balanced_outline": args.balanced_outline,
        "wall_clearance": args.wall_clearance,
        "initial_native_floor_clearance": args.initial_native_floor_clearance,
        "corolla_rotvecs": (
            {str(branch): args.shared_corolla_tilt for branch in range(12)} if args.shared_corolla_tilt else {}
        ),
        "initial_empty_table_regions": (
            [[490, 420, 535, 478]] if args.initial_empty_table_regions else []
        ),
        "initial_roll_seed_deg": args.initial_roll_seed_deg,
        "branch2_diagnostic": args.branch2_diagnostic,
        "secondary_branch_offsets": {
            str(branch): fit.x[20 + 3 * i : 23 + 3 * i].tolist() for i, branch in enumerate(secondary_branches)
        }
        if args.secondary_branches
        else {},
        "secondary_branch_limit": "Branches4/6 were observed projecting into empty tabletop in113diagnostic;113must no longer be treated as untouched model-selection holdout. Other held-out times remain required.",
        "source_corolla_centers": args.source_corolla_centers,
        "seed_fit": str(args.seed_fit) if args.seed_fit else None,
        "initial_detail_view": {
            "enabled": args.initial_detail_view,
            "scope": "113 main-body envelope is fitted, excluding the separate sprig and occluded paper endpoints.110 is a coarse independent initial holdout.",
            "limitation": "113 hand touches paper; approximate static pose must be checked, not assumed from fit quality.",
        },
        "green_wall_envelope": {
            "enabled": args.green_wall_envelope,
            "scope": "Estimated initial-world sprig geometry fixed; release pose must avoid measured cabinet/wall. No contact attraction or dynamic stability objective.",
            "limitations": "Sprig length/leaf geometry remain estimates; this is not a validated complete source mesh.",
        },
        "green_tip_hypothesis": {
            "enabled": args.green_tip_hypothesis,
            "identity": "Unconfirmed: initial111/112 long sprig and release123.4 right-upper sprig may differ. Diagnostic only.",
            "release_observation": {"source_s": 123.4, "pixel": [995, 35], "sigma_px": 12},
            "predictions": {str(t): predict_green_tip(fit.x, t).tolist() for t in times}
            if args.green_tip_hypothesis
            else {},
        },
        "branch2_selected_vertex_count": int(branch_mask.sum()),
        "partial_envelope_limit": "When branch2 diagnostic is enabled, initial tall green sprig is excluded; this cannot validate the complete bouquet.",
        "marker_prediction_minus_observation_px": {str(t): value.tolist() for t, value in common_residual.items()},
        "relative_marker_limit": "When enabled, displayed projection subtracts independently measured common marker image residual; this is a relative-vector diagnostic, not a calibrated new camera or raw reprojection pass.",
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
