"""Clear the free middle finger and the thumb's pre-push fold using actual hand geometry."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np


def main():
    import mujoco
    import trimesh
    from scipy.optimize import least_squares
    from scipy.spatial import ConvexHull
    from scipy.spatial.transform import Rotation

    from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields, refresh_fk

    p = argparse.ArgumentParser()
    for name in ["run", "motion", "out"]:
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    order = a.run.parents[1] / "curated_snapshot/retarget/ground_support_27_end/sidecars/motion_order.json"
    seed = MotionFields.load(a.motion, order)
    robot = Path(
        "/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof_dex3/g1_29dof_dex3_offsethand.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(robot))
    data = mujoco.MjData(model)
    addr = np.array([model.joint(n).qposadr[0] for n in seed.joint_names])
    names = [
        "left_hand_middle_0_joint",
        "left_hand_middle_1_joint",
        "left_hand_thumb_2_joint",
        "left_hand_thumb_1_joint",
    ]
    cols = [seed.joint_names.index(n) for n in names]
    aa = addr[cols]
    q = seed.joint_pos.copy()
    first, last = 400, 1025
    previous = q[first, cols].astype(float)
    endpoint = q[last, cols].astype(float)
    lo = np.array([model.joint(n).range[0] - 1e-7 for n in names])
    hi = np.array([model.joint(n).range[1] + 1e-7 for n in names])
    hand = {
        n: model.geom("left_hand_" + n + "_collision").id
        for n in ["palm", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1"]
    }
    pairs = [
        (hand[mid], g)
        for mid in ["middle_0", "middle_1"]
        for name, g in hand.items()
        if not name.startswith("middle") and name != "palm"
    ] + [(hand["thumb_2"], hand["palm"])]
    visuals = {}
    for name, proxy in hand.items():
        ids = [
            g
            for g in range(model.ngeom)
            if model.geom_bodyid[g] == model.geom_bodyid[proxy]
            and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
            and not model.geom_contype[g]
            and not model.geom_conaffinity[g]
        ]
        if len(ids) == 1:
            visuals[name] = ids[0]
    pairs.extend((visuals[mid], visuals[thumb]) for mid in ["middle_0", "middle_1"] for thumb in ["thumb_1", "thumb_2"])
    points = {}
    for name in ["middle_0", "middle_1", "thumb_1", "thumb_2"]:
        g = hand[name]
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
            half = model.geom_size[g]
            points[g] = np.array(
                [
                    [x, y, z]
                    for x in np.linspace(-half[0], half[0], 5)
                    for y in np.linspace(-half[1], half[1], 5)
                    for z in np.linspace(-half[2], half[2], 5)
                ]
            )
            continue
        mi = model.geom_dataid[g]
        v = model.mesh_vert[model.mesh_vertadr[mi] : model.mesh_vertadr[mi] + model.mesh_vertnum[mi]]
        _, ix = np.unique(np.round(v / 0.001), axis=0, return_index=True)
        points[g] = v[ix]
    for name in ["middle_0", "middle_1", "thumb_1", "thumb_2"]:
        g = visuals[name]
        mi = model.geom_dataid[g]
        v = model.mesh_vert[model.mesh_vertadr[mi] : model.mesh_vertadr[mi] + model.mesh_vertnum[mi]]
        _, ix = np.unique(np.round(v / 0.001), axis=0, return_index=True)
        points[g] = v[ix]
    assets = a.run / "scene/10_flower_bouquet/registered_assets_paper_wrapper_v1"
    objects = {}
    for name in ["flower_bouquet", "small_wicker_basket"]:
        track = np.load(a.run / f"retarget/hoi/10_flower_bouquet/tracks_v8/{name}.npz")
        hulls = []
        for path in sorted(assets.glob(name + "_hull*.obj")):
            v = trimesh.load(path, force="mesh", process=False).vertices
            hulls.append((np.unique(np.round(ConvexHull(v).equations, 10), axis=0), v.min(axis=0), v.max(axis=0)))
        objects[name] = (track, hulls)
    records = []
    future_low = np.tile(lo, (len(q), 1))
    future_high = np.tile(hi, (len(q), 1))
    for k in range(first, last + 1):
        time = 108 + k / 50
        envelope = max(0, min(1, (time - 124) / 0.2, (128.5 - time) / 0.7))
        for j in [2, 3]:
            future_low[k, j] = max(lo[j], seed.joint_pos[k, cols[j]] - 0.35 * envelope)
            future_high[k, j] = min(hi[j], seed.joint_pos[k, cols[j]] + 0.35 * envelope)
    future_low[last] = endpoint
    future_high[last] = endpoint
    for k in range(last - 1, first - 1, -1):
        future_low[k] = np.maximum(future_low[k], future_low[k + 1] - 2.95 / 50)
        future_high[k] = np.minimum(future_high[k], future_high[k + 1] + 2.95 / 50)
    for k in range(first, last + 1):
        time = 108 + k / 50
        data.qpos[:3] = seed.body_pos_w[k, 0]
        data.qpos[3:7] = seed.body_quat_w[k, 0]
        data.qpos[addr] = seed.joint_pos[k]
        mujoco.mj_forward(model, data)
        baseline = q[k, cols].astype(float)
        open_weight = max(0, min(1, time - 116, 128.5 - time))
        finger_reference = baseline.copy()
        finger_reference[:2] *= 1 - open_weight
        push_weight = max(0, min(1, (time - 123.8), (128.5 - time) / 0.7))
        finger_reference[:2] = (1 - push_weight) * finger_reference[:2] + push_weight * np.array([-1.15, -1.0])
        frames = {
            name: (Rotation.from_quat(tr["quaternion_xyzw"][k]).as_matrix(), tr["position_w"][k])
            for name, (tr, _) in objects.items()
        }
        thumb_anchor = None
        if 125 <= time <= 127.8:
            g = visuals["thumb_2"]
            vertices = points[g]
            world = vertices @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
            rot, pos = frames["small_wicker_basket"]
            local = (world - pos) @ rot
            distance = np.min(
                [np.abs(np.max(local @ e[:, :3].T + e[:, 3], axis=1)) for e, _, _ in objects["small_wicker_basket"][1]],
                axis=0,
            )
            ix = np.argmin(distance)
            thumb_anchor = (g, vertices[ix], world[ix])

        def residual(x, frames=frames, baseline=baseline, finger_reference=finger_reference, thumb_anchor=thumb_anchor):
            data.qpos[aa] = x
            mujoco.mj_forward(model, data)
            gaps = np.array([mujoco.mj_geomDistance(model, data, g1, g2, 0.03, None) for g1, g2 in pairs])
            values = list(10000 * np.maximum(0.005 - gaps, 0))
            for g, v in points.items():
                world = v @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
                for name, (_, hulls) in objects.items():
                    rot, pos = frames[name]
                    local = (world - pos) @ rot
                    dist = []
                    for e, low, high in hulls:
                        near = np.all((local >= low - 0.005) & (local <= high + 0.005), axis=1)
                        signed = np.full(len(local), 0.005)
                        signed[near] = np.max(local[near] @ e[:, :3].T + e[:, 3], axis=1)
                        dist.append(signed)
                    values.extend(30000 * np.maximum(0.0002 - np.min(dist, axis=0), 0))
            values.extend(5 * (x - finger_reference))
            if thumb_anchor is not None:
                g, vertex, target = thumb_anchor
                delta = vertex @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g] - target
                values.append(20000 * max(0, np.linalg.norm(delta) - 0.001))
            return np.asarray(values)

        lower = np.maximum(future_low[k], previous - 2.95 / 50)
        upper = np.minimum(future_high[k], previous + 2.95 / 50)
        if k in [first, last]:
            result = baseline
        else:
            eq = lower >= upper
            if np.any(eq):
                if np.max((lower - upper)[eq]) > 1e-6:
                    raise ValueError((time, "infeasible finger bounds", lower - upper))
                mid = (lower[eq] + upper[eq]) / 2
                lower[eq] = mid - 1e-8
                upper[eq] = mid + 1e-8
            free = upper - lower > 1e-6
            fixed = (lower + upper) / 2

            def reduced(y, fixed=fixed, free=free, residual=residual):
                x = fixed.copy()
                x[free] = y
                return residual(x)

            result = fixed.copy()
            if np.any(free):
                result[free] = least_squares(
                    reduced,
                    np.clip(previous[free], lower[free], upper[free]),
                    bounds=(lower[free], upper[free]),
                    max_nfev=80,
                ).x
        q[k, cols] = result
        previous = result.copy()
        residual(result)
        records.append(
            {
                "source_s": time,
                "min_native_gap_m": min(
                    float(mujoco.mj_geomDistance(model, data, g1, g2, 0.03, None)) for g1, g2 in pairs
                ),
                "joint_delta_rad": (result - baseline).tolist(),
            }
        )
        if k % 25 == 0:
            (a.out / "progress.json").write_text(json.dumps(records[-1]))
            np.save(a.out / "joint_pos_checkpoint.npy", q)
    refresh_fk(replace(seed, joint_pos=q), robot).save(a.out / "motion.npz")
    (a.out / "report.json").write_text(
        json.dumps(
            {"status": "unreviewed candidate; middle contact not assumed", "finger_names": names, "records": records},
            indent=2,
        )
    )
    print(a.out)


if __name__ == "__main__":
    main()
