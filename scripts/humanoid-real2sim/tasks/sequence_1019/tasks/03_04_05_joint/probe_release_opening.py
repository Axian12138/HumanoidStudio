"""Search velocity-reachable opening poses with the actual swept hat geometry."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    import mujoco
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation, Slerp

    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--seed-json", type=Path)
    p.add_argument("--count", type=int, default=16)
    p.add_argument("--rack-clearance", type=float, default=-0.001)
    p.add_argument("--output-name", default="03_04_05_release_opening_family.json")
    p.add_argument("--candidate", default="03_04_05_joint_v68")
    p.add_argument("--start", type=float, default=47.64)
    p.add_argument("--end", type=float, default=47.76)
    p.add_argument("--extrapolate-seed", action="store_true")
    p.add_argument("--two-knots", action="store_true")
    a = p.parse_args()
    src = a.run / "retarget/candidates" / a.candidate
    motion = np.load(src / "motion.npz")
    hat = np.load(src / "black_ribbon_hat.npz")
    names = json.loads((src / "motion_order.json").read_text())["robot_joint_names"]
    arm = [
        "right_" + n + "_joint"
        for n in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
    ]
    selected = arm + [n for n in names if n.startswith("right_hand")]
    cols = [names.index(n) for n in selected]
    model = mujoco.MjModel.from_xml_path(
        str(a.run / "scene/03_04_05/robot_props_drawer_with_rack_early_registered.xml")
    )
    data = mujoco.MjData(model)
    addr = np.array([model.joint(n).qposadr[0] for n in names])
    ha = model.joint("black_ribbon_hat_free").qposadr[0]
    hb = model.body("black_ribbon_hat").id
    rb = model.body("coat_rack_support").id
    hand = [
        i
        for i in range(model.nbody)
        if model.body(i).name.startswith("right_hand") or model.body(i).name == "right_wrist_yaw_link"
    ]
    handmap = {b: i for i, b in enumerate(hand)}
    for gi in range(model.ngeom):
        if model.geom_bodyid[gi] not in [*hand, hb, rb]:
            model.geom_contype[gi] = 0
            model.geom_conaffinity[gi] = 0
        elif a.rack_clearance > 0 and model.geom_bodyid[gi] == rb:
            model.geom_margin[gi] = a.rack_clearance + 0.001
    start, end = a.start, a.end
    ts = np.linspace(start, end, round((end - start) * 100) + 1)
    time = 27 + np.arange(len(motion["joint_pos"])) / 50
    k0, k1 = round((start - 27) * 50), round((end - 27) * 50)
    prior = motion["joint_pos"][k1, cols]
    release = motion["joint_pos"][k0, cols]
    limits = np.array([37] * 5 + [22] * 2 + [3] * 7)
    low = np.maximum([model.joint(n).range[0] for n in selected], release - limits * (end - start))
    high = np.minimum([model.joint(n).range[1] for n in selected], release + limits * (end - start))
    if a.two_knots:
        low, high = np.tile(low, 2), np.tile(high, 2)
    rootq = Slerp(time, Rotation.from_quat(motion["body_quat_w"][:, 0][:, [1, 2, 3, 0]]))(ts).as_quat()
    hatq = Slerp(time, Rotation.from_quat(hat["quaternion_xyzw"]))(ts).as_quat()

    def depths(x):
        results = np.zeros((len(ts), len(hand) * 2))
        for si, t in enumerate(ts):
            f = (t - 27) * 50
            k = int(np.floor(f))
            u = f - k

            def blend(array, k=k, u=u):
                return array[k] * (1 - u) + array[k + 1] * u

            data.qpos[:3] = blend(motion["body_pos_w"][:, 0])
            data.qpos[3:7] = rootq[si][[3, 0, 1, 2]]
            data.qpos[addr] = blend(motion["joint_pos"])
            alpha = (t - start) / (end - start)
            if a.two_knots:
                first, second = x.reshape(2, len(cols))
                data.qpos[addr[cols]] = (
                    (release * (1 - 2 * alpha) + first * (2 * alpha))
                    if alpha <= 0.5
                    else (first * (2 - 2 * alpha) + second * (2 * alpha - 1))
                )
            else:
                data.qpos[addr[cols]] = release * (1 - alpha) + x * alpha
            data.qpos[ha : ha + 3] = blend(hat["position_w"])
            data.qpos[ha + 3 : ha + 7] = hatq[si][[3, 0, 1, 2]]
            mujoco.mj_forward(model, data)
            for c in data.contact:
                b1, b2 = map(int, model.geom_bodyid[[c.geom1, c.geom2]])
                for target, offset in ((hb, 0), (rb, len(hand))):
                    if target in (b1, b2):
                        other = b2 if b1 == target else b1
                        if other in handmap:
                            j = offset + handmap[other]
                            results[si, j] = max(
                                results[si, j], -c.dist + (a.rack_clearance if target == rb else -0.001)
                            )
        return results

    rng = np.random.default_rng(71)
    records = []
    seed_pose = np.array(json.loads(a.seed_json.read_text())["joint_pos"]) if a.seed_json else prior
    if a.extrapolate_seed:
        seed_pose = release + (release - motion["joint_pos"][k0 - 6, cols]) * ((end - start) / 0.12)
    if a.two_knots:
        seed_pose = np.r_[motion["joint_pos"][round((k0 + k1) / 2), cols], seed_pose]
        prior_vector = np.r_[motion["joint_pos"][round((k0 + k1) / 2), cols], prior]
    else:
        prior_vector = prior
    for seed in range(a.count):
        initial = np.clip(
            seed_pose if seed == 0 else np.tile(release, 2 if a.two_knots else 1) + rng.normal(0, 0.3, len(low)),
            low + 1e-7,
            high - 1e-7,
        )

        def residual(x):
            if a.two_knots:
                velocity_excess = np.maximum(
                    np.abs(np.diff(np.vstack([release, x.reshape(2, len(cols))]), axis=0)) / ((end - start) / 2)
                    - limits,
                    0,
                )
            else:
                velocity_excess = []
            return np.r_[depths(x).ravel() * 4000, (x - prior_vector) * 0.03, np.asarray(velocity_excess).ravel() * 500]

        fit = least_squares(residual, initial, bounds=(low, high), max_nfev=150)
        record = {
            "seed": seed,
            "joint_pos": fit.x.tolist(),
            "max_excess_m": float(depths(fit.x).max()),
            "cost": float(fit.cost),
            "success": bool(fit.success),
            "two_knots": a.two_knots,
        }
        records.append(record)
        print(record, flush=True)
    path = a.run / "review/scene" / a.output_name
    path.write_text(
        json.dumps(
            {
                "status": "opening_sweep_search_not_accepted",
                "times_s": [start, end],
                "joint_names": selected,
                "rack_clearance_m": a.rack_clearance,
                "records": records,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
