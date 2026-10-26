"""Receding-window body prior using retarget3's existing geometry constraints.

The external retarget3 package supplies FK and per-frame objectives. All stance
anchors and joint speed bounds remain hard constraints across window seams.
"""

import json

import numpy as np


def solve_window_prior(optimizer, data, output, *, horizon=60, commit=20):
    import casadi as ca

    if not 1 <= commit < horizon:
        raise ValueError("require 1 <= commit < horizon")
    if optimizer.scene is not None:
        raise ValueError("window body prior currently supports floor/self guards only")
    solver = optimizer
    robot = solver.robot
    seed = data["qsource"]
    n = len(seed)
    prior = np.array([solver.packx(row) for row in seed])
    prior[:, 5] = np.unwrap(prior[:, 5])
    vmax = np.r_[[0.025] * 3, [0.08] * 3, np.minimum(robot.v_abs[solver.ids - 1], 6.0) / solver.fps]
    # Compile one fixed horizon. The final window repeats its last valid frame;
    # repeated frames are optimization padding, never exported motion.
    x = ca.MX.sym("window", 35, horizon)
    parameter_size = int(solver.gfun.size_in(1)[0])
    p = ca.MX.sym("parameters", parameter_size, horizon)
    previous = ca.MX.sym("previous", 35)
    first = ca.MX.sym("first")
    constraints, lower_g, upper_g = [], [], []
    cost = 0
    frame_cost = solver.solver.get_function("nlp_f")
    for j in range(horizon):
        cost += frame_cost(x[:, j], p[:, j])
        constraints.append(solver.gfun(x[:, j], p[:, j]))
        lower_g.extend(solver.lo)
        upper_g.extend(solver.hi)
        delta = x[:, j] - (x[:, j - 1] if j else previous)
        active = 1 if j else first
        constraints.append(active * delta)
        lower_g.extend(-vmax)
        upper_g.extend(vmax)
        cost += 8 * ca.sumsqr(active * delta)
        if j > 1:
            cost += 2 * ca.sumsqr(x[:, j] - 2 * x[:, j - 1] + x[:, j - 2])
    window = ca.nlpsol(
        "body_window",
        "ipopt",
        {"x": ca.vec(x), "p": ca.vertcat(ca.vec(p), previous, first), "f": cost, "g": ca.vertcat(*constraints)},
        {"print_time": False, "ipopt": {"print_level": 0, "max_iter": 250, "tol": 1e-6, "bound_relax_factor": 0}},
    )
    params = []
    for k in range(n):
        params.append(
            np.r_[
                seed[k],
                prior[k],
                prior[max(0, k - 1)],
                data["target"][k, :15].ravel(),
                data["torso"][k].ravel(order="F"),
                data["feet"][k].ravel(),
                data["feet_rot"][k, 0].ravel(order="F"),
                data["feet_rot"][k, 1].ravel(order="F"),
                data["contacts"][k].astype(float),
                data["hand_pos"][k].ravel(),
                data["hand_rot"][k, 0].ravel(order="F"),
                data["hand_rot"][k, 1].ravel(order="F"),
                [0, 1, 0, 0],
            ]
        )
    params = np.asarray(params)
    solved = []
    reports = []
    warm = None
    for start in range(0, n, commit):
        indices = np.minimum(np.arange(start, start + horizon), n - 1)
        ps = params[indices].T
        initial = prior[indices].T.copy()
        if warm is not None:
            initial[:, : horizon - commit] = warm[:, commit:]
        prev = solved[-1] if solved else prior[0]
        pp = np.r_[ps.ravel(order="F"), prev, float(start > 0)]
        result = window(
            x0=initial.ravel(order="F"),
            p=pp,
            lbx=np.tile(solver.lower, horizon),
            ubx=np.tile(solver.upper, horizon),
            lbg=lower_g,
            ubg=upper_g,
        )
        values = np.array(result["g"]).ravel()
        violation = float(max(0, np.max(np.asarray(lower_g) - values), np.max(values - np.asarray(upper_g))))
        status = window.stats()["return_status"]
        warm = np.array(result["x"]).reshape(35, horizon, order="F")
        reports.append({"start": start, "status": status, "violation": violation})
        if not window.stats()["success"] or violation > 2e-5:
            np.savez_compressed(
                output / "intermediate/failed_window.npz", start=start, x=warm, parameters=ps, previous=prev
            )
            (output / "review/window_solver.json").write_text(json.dumps(reports, indent=2))
            raise RuntimeError(f"Window {start}: {status}, violation={violation}")
        solved.extend(warm[:, : min(commit, n - start)].T.copy())
        np.save(output / "intermediate/window_x_checkpoint.npy", np.array(solved))
        print("window", start, "/", n, status, "violation", violation, flush=True)
    q = np.array([np.array(solver.qfun(row, params[k])).ravel() for k, row in enumerate(solved)])
    np.save(output / "intermediate/body_prior_q.npy", q)
    (output / "review/window_solver.json").write_text(json.dumps(reports, indent=2))
    return q
