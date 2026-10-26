"""Rebuild flower web grasp and two-finger basket hook with a bounded wrist pitch."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

ROBOT = Path(
    "/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof_dex3/g1_29dof_dex3_offsethand.xml"
)


def smooth(t):
    t = np.clip(t, 0, 1)
    return t**3 * (10 - 15 * t + 6 * t * t)


def main():
    import mujoco
    import trimesh
    from scipy.optimize import least_squares
    from scipy.spatial import ConvexHull
    from scipy.spatial.transform import Rotation as R

    from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields, refresh_fk

    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=Path)
    p.add_argument("--only-side", choices=["left", "right"])
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    parent = json.load(open(a.run / "inputs/parent.json"))
    info = parent["parent"]
    seed = MotionFields.load(a.seed or parent["motion"], info["order"])
    q = seed.joint_pos.copy()
    m = mujoco.MjModel.from_xml_path(str(ROBOT))
    d = mujoco.MjData(m)
    m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    addr = np.array([m.joint(n).qposadr[0] for n in seed.joint_names])
    design = json.load(open(a.run / "review/grip_design.json"))[0]
    center = np.array(design["center"])
    tracks = {n: np.load(Path(info["tracks"]) / (n + ".npz")) for n in ["flower_bouquet", "small_wicker_basket"]}
    hulls = {}
    for n in tracks:
        hulls[n] = [
            np.unique(np.round(ConvexHull(trimesh.load(f, force="mesh", process=False).vertices).equations, 8), axis=0)
            for f in sorted(Path(info["assets"]).glob(n + "_hull*.obj"))
        ]
    samples = {}
    for g in range(m.ngeom):
        name = m.geom(g).name or ""
        if not name.endswith("_collision") or not any(x in name for x in ["hand_", "wrist_collision"]):
            continue
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mi = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[mi] : m.mesh_vertadr[mi] + m.mesh_vertnum[mi]].copy()
            _, ix = np.unique(np.round(v / 0.007), axis=0, return_index=True)
            v = v[ix]
        elif m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
            s = m.geom_size[g]
            v = np.array([[x, y, z] for x in [-s[0], 0, s[0]] for y in [-s[1], 0, s[1]] for z in [-s[2], 0, s[2]]])
        else:
            continue
        samples[g] = (name, v)

    def pose(k, qq):
        d.qpos[:3] = seed.body_pos_w[k, 0]
        d.qpos[3:7] = seed.body_quat_w[k, 0]
        d.qpos[addr] = qq
        mujoco.mj_forward(m, d)

    reports = []
    for side, begin, contact, release, end in [
        ("right", 111.0, 113.2, 123.2, 125.8),
        ("left", 116.2, 118.7, 120.4, 122.2),
    ]:
        if a.only_side and side != a.only_side:
            continue
        cols = np.array(
            [
                seed.joint_names.index(side + "_" + n + "_joint")
                for n in [
                    "shoulder_pitch",
                    "shoulder_roll",
                    "shoulder_yaw",
                    "elbow",
                    "wrist_roll",
                    "wrist_pitch",
                    "wrist_yaw",
                ]
            ]
        )
        fc = np.array([seed.joint_names.index(side + "_hand_" + n + "_joint") for n in design["names"]])
        wb = m.body(side + "_wrist_yaw_link").id
        lo, hi = np.array([m.joint(seed.joint_names[c]).range for c in cols]).T
        first, last = [round((v - 108) * 50) for v in [begin, end]]
        previous = q[first, cols].astype(float)
        rows = []
        pairnames = [(side + "_shoulder_yaw_collision", "torso_collision" + str(i)) for i in [1, 2, 3]] + [
            (side + "_elbow_yaw_collision", "torso_collision2"),
            (side + "_elbow_yaw_collision", side + "_wrist_collision"),
        ]
        pairs = [(m.geom(x).id, m.geom(y).id) for x, y in pairnames]
        for k in range(first, last + 1):
            t = 108 + k / 50
            w = min(smooth((t - begin) / (contact - begin)), smooth((end - t) / (end - release)))
            pose(k, seed.joint_pos[k])
            oldp = d.xpos[wb].copy()
            oldr = d.xmat[wb].reshape(3, 3).copy()
            source_gaps = np.array([mujoco.mj_geomDistance(m, d, g, h, 0.02, None) for g, h in pairs])
            frame = {
                n: (R.from_quat(tr["quaternion_xyzw"][k]).as_matrix(), tr["position_w"][k]) for n, tr in tracks.items()
            }
            if side == "right":
                rr, pp = frame["flower_bouquet"]
                target_center = pp + rr @ np.array([0, 0, 0.25])
                axis = rr[:, 2]
                target = (1 - w) * (oldp + oldr @ center) + w * target_center
                targetaxis = (1 - w) * oldr[:, 2] + w * axis
                targetaxis /= np.linalg.norm(targetaxis)
                grip = np.array(design["hand"])
                openhand = np.zeros(7)
                if t < 112.5:
                    hand = openhand
                elif t <= 123.2:
                    hand = smooth((t - 112.5) / 0.7) * grip
                else:
                    hand = (1 - smooth((t - 123.2) / 0.45)) * grip
                # Keep the original source-timed retreat, with the grasp offset decaying.
                if t > 123.2:
                    kr = 760
                    pose(kr, seed.joint_pos[kr])
                    oldrelease = d.xpos[wb] + d.xmat[wb].reshape(3, 3) @ center
                    r0, p0 = (
                        R.from_quat(tracks["flower_bouquet"]["quaternion_xyzw"][kr]).as_matrix(),
                        tracks["flower_bouquet"]["position_w"][kr],
                    )
                    offset = p0 + r0 @ np.array([0, 0, 0.25]) - oldrelease
                    target = oldp + oldr @ center + w * offset
                q[k, fc] = (1 - w) * seed.joint_pos[k, fc] + w * hand
            else:
                rr, pp = frame["small_wicker_basket"]
                # Both proximal finger joints remain extended; distal joints hook just below the rim.
                hand = np.array([0, 0.25, 0.3, 0, -0.55, 0, -0.55])
                q[k, fc] = (1 - w) * seed.joint_pos[k, fc] + w * hand
                pose(k, q[k])
                tip = [m.body("left_hand_" + n + "_finger_tip").id for n in ["index", "middle"]]
                localtip = np.mean([(d.xpos[i] - d.xpos[wb]) @ d.xmat[wb].reshape(3, 3) for i in tip], axis=0)
                center = localtip
                target = (1 - w) * (oldp + oldr @ center) + w * (pp + rr @ np.array([0, -0.157, 0.192]))
                targetaxis = (1 - w) * oldr[:, 2] + w * rr[:, 0]
                targetaxis /= np.linalg.norm(targetaxis)
            pose(k, q[k])

            def fun(
                x,
                cols=cols,
                wb=wb,
                center=center,
                side=side,
                frame=frame,
                pairs=pairs,
                target=target,
                targetaxis=targetaxis,
                w=w,
                previous=previous,
                k=k,
                source_gaps=source_gaps,
            ):
                d.qpos[addr[cols]] = x
                mujoco.mj_forward(m, d)
                wr = d.xmat[wb].reshape(3, 3)
                pos = d.xpos[wb] + wr @ center
                coll = []
                for g, (name, v) in samples.items():
                    if not name.startswith(side + "_"):
                        continue
                    world = v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
                    for n, eqs in hulls.items():
                        rr, pp = frame[n]
                        local = (world - pp) @ rr
                        dist = np.min([np.max(local @ e[:, :3].T + e[:, 3], axis=1) for e in eqs], axis=0)
                        coll.extend(np.minimum(dist - 0.0003, 0))
                gaps = np.array([mujoco.mj_geomDistance(m, d, g, h, 0.02, None) for g, h in pairs])
                return np.r_[
                    (pos - target) * 500,
                    (wr[:, 2] - targetaxis) * (100 if side == "left" else 25),
                    x[5] * 5 * w,
                    (x - previous) * 1.5,
                    (x - seed.joint_pos[k, cols]) * 0.15,
                    np.asarray(coll) * 1800,
                    np.minimum(gaps - np.minimum(source_gaps, 0.002), 0) * 1500,
                ]

            lower = lo.copy()
            upper = hi.copy()
            bound = (1 - w) * max(abs(float(seed.joint_pos[k, cols[5]])), 0.44) + w * 0.44
            lower[5] = max(lower[5], -bound)
            upper[5] = min(upper[5], bound)
            x0 = np.clip(previous, lower + 1e-8, upper - 1e-8)
            fit = least_squares(fun, x0, bounds=(lower, upper), max_nfev=70, diff_step=1e-4, ftol=1e-6)
            q[k, cols] = fit.x
            previous = fit.x
            res = fun(fit.x)
            rows.append([t, float(np.linalg.norm(res[:3]) / 500), float(fit.x[5]), float(fit.cost)])
            if k % 100 == 0:
                print(side, t, rows[-1], flush=True)
        reports.append({"side": side, "rows": rows})
        center = np.array(design["center"])
    refresh_fk(replace(seed, joint_pos=q), ROBOT).save(a.out / "motion.npz")
    (a.out / "motion_order.json").write_bytes(Path(info["order"]).read_bytes())
    (a.out / "report.json").write_text(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
