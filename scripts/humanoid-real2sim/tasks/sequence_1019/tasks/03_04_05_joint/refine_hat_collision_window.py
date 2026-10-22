"""Refine arm and fingers against authored hat collisions with fixed window endpoints."""

import argparse
import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np

from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields, refresh_fk

MODEL = Path(
    "/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof_dex3/g1_29dof_dex3_offsethand.xml"
)


def main():
    import mujoco
    from scipy.interpolate import CubicSpline
    from scipy.optimize import least_squares

    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--input-version", required=True)
    p.add_argument("--output-version", required=True)
    p.add_argument("--start", type=float, required=True)
    p.add_argument("--end", type=float, required=True)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--velocity-limit", type=float, default=0)
    p.add_argument("--named-velocity-limits", action="store_true")
    p.add_argument("--sofa", action="store_true")
    p.add_argument("--rack", action="store_true")
    p.add_argument("--fast-release-knots", action="store_true")
    p.add_argument("--midpoints", action="store_true")
    p.add_argument("--fingers-only", action="store_true")
    p.add_argument("--grasp-start", type=float)
    a = p.parse_args()
    src = a.run / "retarget/candidates" / a.input_version
    out = a.run / "retarget/candidates" / a.output_version
    out.mkdir(exist_ok=False)
    motion = MotionFields.load(src / "motion.npz", src / "motion_order.json")
    hat = np.load(src / "black_ribbon_hat.npz")
    model_path = a.run / "scene/03_04_05/robot_props_drawer_with_rack_early_registered.xml"
    if a.sofa:
        import copy
        import xml.etree.ElementTree as ET

        tree = ET.parse(model_path)
        support = ET.parse(a.run / "scene/03_04_05/hat_registered_sofa_physics.xml")
        tree.getroot().find("asset").append(copy.deepcopy(support.getroot().find(".//hfield")))
        tree.getroot().find("worldbody").append(copy.deepcopy(support.getroot().find(".//geom[@type='hfield']")))
        model_path = out / "sofa_collision_model.xml"
        tree.write(model_path)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    d = mujoco.MjData(model)
    arm = [
        "right_" + n + "_joint"
        for n in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
    ]
    names = ([] if a.fingers_only else arm) + [n for n in motion.joint_names if n.startswith("right_hand")]
    assert len(arm) == 7 and all("leg" not in n and "ankle" not in n for n in names)
    cols = [motion.joint_names.index(n) for n in names]
    addr = np.array([model.joint(n).qposadr[0] for n in motion.joint_names])
    hb = model.body("black_ribbon_hat").id
    hand = [model.body(n).id for n in motion.body_names if n.startswith("right_hand") or n == "right_wrist_yaw_link"]
    handmap = {b: i for i, b in enumerate(hand)}
    hat_geoms = [i for i in range(model.ngeom) if model.geom_bodyid[i] == hb and model.geom_contype[i]]
    grasp_geoms = [
        [i for i in range(model.ngeom) if model.geom_bodyid[i] == model.body(n).id and model.geom_contype[i]]
        for n in ["right_hand_thumb_2_link", "right_hand_index_1_link"]
    ]
    # Other scene objects remain excluded from this local objective, and are audited afterwards.
    rb = model.body("coat_rack_support").id
    allowed = {*hand, hb}
    if a.rack:
        allowed.add(rb)
    for gi in range(model.ngeom):
        if int(model.geom_bodyid[gi]) not in allowed and not (a.sofa and model.geom(gi).name == "registered_sofa_top"):
            model.geom_contype[gi] = 0
            model.geom_conaffinity[gi] = 0
    sofa_vertices = []
    if a.sofa:
        from scipy.interpolate import RegularGridInterpolator

        sg = model.geom("registered_sofa_top").id
        hf = int(model.geom_dataid[sg])
        nr, nc = int(model.hfield_nrow[hf]), int(model.hfield_ncol[hf])
        start = int(model.hfield_adr[hf])
        size = model.hfield_size[hf]
        height = model.hfield_data[start : start + nr * nc].reshape(nr, nc) * size[2] + model.geom_pos[sg, 2]
        sofa_height = RegularGridInterpolator(
            (
                np.linspace(-size[1], size[1], nr) + model.geom_pos[sg, 1],
                np.linspace(-size[0], size[0], nc) + model.geom_pos[sg, 0],
            ),
            height,
            bounds_error=False,
            fill_value=-np.inf,
        )
        for gi in range(model.ngeom):
            body = int(model.geom_bodyid[gi])
            if body in handmap and model.geom_contype[gi] and model.geom_type[gi] == mujoco.mjtGeom.mjGEOM_MESH:
                mesh_id = int(model.geom_dataid[gi])
                va = int(model.mesh_vertadr[mesh_id])
                vn = int(model.mesh_vertnum[mesh_id])
                sofa_vertices.append((gi, handmap[body], model.mesh_vert[va : va + vn].copy()))
    q = motion.joint_pos.astype(float).copy()
    time = 27 + np.arange(len(q)) / 50
    rows = np.flatnonzero((time >= a.start) & (time <= a.end))
    sample = np.unique(np.r_[np.arange(0, len(rows), a.stride), len(rows) - 1])
    if a.midpoints:
        sample = np.arange(0, len(rows) - 0.5, 0.5)
    from scipy.spatial.transform import Rotation, Slerp

    sample_times = a.start + sample / 50
    root_rot = Slerp(time, Rotation.from_quat(motion.body_quat_w[:, 0][:, [1, 2, 3, 0]]))(sample_times).as_quat()
    hat_rot = Slerp(time, Rotation.from_quat(hat["quaternion_xyzw"]))(sample_times).as_quat()
    knots = (
        a.start + (a.end - a.start) * np.array([0, 0.04, 0.10, 0.22, 0.45, 0.7, 1])
        if a.fast_release_knots
        else np.linspace(a.start, a.end, 7)
    )
    lo = np.array([model.joint(n).range[0] for n in names])
    hi = np.array([model.joint(n).range[1] for n in names])
    velocity_limits = np.full(len(names), a.velocity_limit)
    if a.named_velocity_limits:
        import xml.etree.ElementTree as ET

        limits = {
            j.get("name"): float(j.find("limit").get("velocity", "20"))
            for j in ET.parse("/home/galbot/workspace/ScaleTrack/retarget3/assets/robot/body.urdf").findall("joint")
            if j.find("limit") is not None
        }
        velocity_limits = np.array([limits.get(n, 3.0) for n in names])
    ha = model.joint("black_ribbon_hat_free").qposadr[0]
    count = 0

    def values(x):
        return q[np.ix_(rows, cols)] + CubicSpline(
            knots, np.vstack([np.zeros(len(cols)), x.reshape(5, len(cols)), np.zeros(len(cols))]), bc_type="clamped"
        )(time[rows])

    def depths(v):
        ndepth = len(hand) * (1 + int(a.sofa) + int(a.rack))
        result = np.zeros((len(sample), ndepth + (2 if a.grasp_start is not None else 0)))
        for si, ri in enumerate(sample):
            i = int(ri)
            j = min(i + 1, len(rows) - 1)
            fraction = ri - i
            k, k1 = rows[i], rows[j]

            def blend(array, k=k, k1=k1, fraction=fraction):
                return array[k] * (1 - fraction) + array[k1] * fraction

            d.qpos[:3] = blend(motion.body_pos_w[:, 0])
            d.qpos[3:7] = root_rot[si][[3, 0, 1, 2]]
            d.qpos[addr] = blend(q)
            d.qpos[addr[cols]] = v[i] * (1 - fraction) + v[j] * fraction
            d.qpos[ha : ha + 3] = blend(hat["position_w"])
            d.qpos[ha + 3 : ha + 7] = hat_rot[si][[3, 0, 1, 2]]
            mujoco.mj_forward(model, d)
            for contact in d.contact:
                b1, b2 = map(int, model.geom_bodyid[[contact.geom1, contact.geom2]])
                rack_contact = a.rack and rb in (b1, b2)
                support_contact = (a.sofa and 0 in (b1, b2)) or rack_contact
                if hb not in (b1, b2) and not support_contact:
                    continue
                target_body = rb if rack_contact else (0 if support_contact else hb)
                other = b2 if b1 == target_body else b1
                if other in handmap:
                    j = handmap[other] + (
                        len(hand) * (1 + int(a.sofa)) if rack_contact else (len(hand) if support_contact else 0)
                    )
                    result[si, j] = max(result[si, j], -float(contact.dist) + (0.002 if support_contact else -0.001))
            if a.sofa:
                for gi, body_index, vertices in sofa_vertices:
                    world = vertices @ d.geom_xmat[gi].reshape(3, 3).T + d.geom_xpos[gi]
                    excess = sofa_height(world[:, [1, 0]]) + 0.002 - world[:, 2]
                    result[si, len(hand) + body_index] = max(result[si, len(hand) + body_index], float(excess.max()))
            if a.grasp_start is not None and time[k] >= a.grasp_start:
                for j, tips in enumerate(grasp_geoms):
                    distance = min(
                        mujoco.mj_geomDistance(model, d, tip, hat_geom, 0.05, None)
                        for tip in tips
                        for hat_geom in hat_geoms
                    )
                    result[si, ndepth + j] = max(0, distance - 0.002) * 0.5
        return result

    prior = q[np.ix_(rows, cols)]
    before = depths(prior)

    def residual(x):
        nonlocal count
        v = values(x)
        dep = depths(v)
        count += 1
        if count % 100 == 0:
            print({"evaluations": count, "max_penetration_excess_m": float(dep.max())}, flush=True)
        return np.r_[
            (dep * 1500).ravel(),
            ((v - prior) * 0.5).ravel(),
            (np.diff(v - prior, n=2, axis=0) * 80).ravel(),
            (np.maximum(lo - v, 0) * 400).ravel(),
            (np.maximum(v - hi, 0) * 400).ravel(),
            (
                np.maximum(abs(np.diff(v, axis=0)) * 50 - velocity_limits, 0)
                * (20 if a.velocity_limit or a.named_velocity_limits else 0)
            ).ravel(),
        ]

    fit = least_squares(residual, np.zeros(5 * len(cols)), max_nfev=60, ftol=1e-5, diff_step=1e-4)
    v = np.clip(values(fit.x), lo, hi)
    after = depths(v)
    result_q = q.copy()
    result_q[np.ix_(rows, cols)] = v
    result = refresh_fk(replace(motion, joint_pos=result_q), MODEL)
    for f in src.iterdir():
        if (
            f.is_file()
            and f.name not in ["motion.npz", "review.json", "manifest.json"]
            and not f.name.startswith("collision_")
        ):
            shutil.copy2(f, out / f.name)
    result.save(out / "motion.npz")
    report = {
        "status": "collision_refinement_pending_full_audit",
        "window_s": [a.start, a.end],
        "sample_stride": a.stride,
        "linear_q_slerp_pose_midpoints": a.midpoints,
        "registered_sofa_patch": a.sofa,
        "registered_rack_constraint": a.rack,
        "grasp_contact_start_s": a.grasp_start,
        "optimized_joints": names,
        "before_max_excess_m": float(before.max()),
        "after_max_excess_m": float(after.max()),
        "maximum_joint_change_rad": float(abs(v - prior).max()),
        "endpoint_delta_rad": float(abs((v - prior)[[0, -1]]).max()),
        "solver_success": bool(fit.success),
        "nfev": fit.nfev,
        "velocity_limit_rad_s": velocity_limits.tolist() if a.named_velocity_limits else a.velocity_limit or None,
        "velocity_contract": "body.urdf named limits; unlisted Dex3 planning fallback3"
        if a.named_velocity_limits
        else "optional uniform optimization regularizer, not hardware rating",
        "maximum_segment_velocity_rad_s": float(abs(np.diff(v, axis=0)).max() * 50),
        "limitations": [
            "Hat and registered sofa patch sampled contact objective"
            if a.sofa
            else "Hat-only sampled contact objective",
            "Full-rate all-object and room audit pending",
            "Object track unchanged",
        ],
    }
    review = json.loads((src / "review.json").read_text())
    review["collision_window_fit"] = report
    review["status"] = report["status"]
    (out / "review.json").write_text(json.dumps(review, indent=2) + "\n")
    print(report)


if __name__ == "__main__":
    main()
