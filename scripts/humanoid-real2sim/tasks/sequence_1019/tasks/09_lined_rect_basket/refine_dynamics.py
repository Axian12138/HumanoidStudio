"""Opt-in offline sampling refinement of the basket's v21 reference.

Uses the unchanged RSS Stage 4 scene, without loading or changing its tracker.
Run with the repository Python and --candidate /path/to/hoi_v21.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from robosimstudio.humanoid_real2sim.paths import run_dir
from robosimstudio.humanoid_real2sim.refinement.export import export_motion
from robosimstudio.humanoid_real2sim.refinement.sampling import Config, optimize
from robosimstudio.humanoid_real2sim.refinement.warp_backend import WarpBackend
from robosimstudio.humanoid_real2sim.track.launcher import (
    DEFAULT_METADATA,
    DEFAULT_ORDER,
    DEFAULT_ROBOT,
    _load_scene_file,
    _position_asset,
)
from robosimstudio.humanoid_real2sim.track.timeline import ReferenceMotion


def match(model, kind, suffix):
    count = model.njnt if kind == "joint" else model.nbody
    hits = [i for i in range(count) if getattr(model, kind)(i).name.endswith(suffix)]
    if len(hits) != 1:
        raise ValueError(f"ambiguous {kind} {suffix}: {hits}")
    return hits[0]


def prepare(candidate, out):
    import mujoco

    metadata = json.loads(DEFAULT_METADATA.read_text())
    motion = ReferenceMotion.load(
        candidate / "motion.npz",
        motion_order=DEFAULT_ORDER,
        joint_names=tuple(metadata["joint_names"]),
        body_names=tuple(metadata["selected_body_names"]),
    )
    temporary, asset = _position_asset(DEFAULT_ROBOT, metadata)
    os.environ["RSS_BASKET_STAGE4_CANDIDATE"] = str(candidate)
    sim = _load_scene_file(Path(__file__).with_name("scene_stage4.py"), asset, motion)
    model = sim.pipeline.solver.mj_model
    model.opt.timestep = 0.002
    model.opt.ls_iterations = 50
    mujoco.mj_saveModel(model, str(out / "scene.mjb"))
    order = json.loads(DEFAULT_ORDER.read_text())
    names = order["robot_joint_names"]
    body_names = order["robot_body_names"]
    with np.load(candidate / "motion.npz") as data:
        joint_pos = data["joint_pos"].copy()
        root_pos = data["body_pos_w"][:, body_names.index("pelvis")].copy()
        root_quat = data["body_quat_w"][:, body_names.index("pelvis")].copy()
    with np.load(candidate / "object_estimate.npz") as data:
        obj_pos, obj_quat = data["position_w"].copy(), data["quaternion_xyzw"].copy()
    q = np.tile(model.qpos0, (motion.frames, 1))
    root_joint = int(np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)[0])
    root_adr = int(model.jnt_qposadr[root_joint])
    obj_body = match(model, "body", "lined_rect_basket")
    obj_adr = int(model.jnt_qposadr[model.body_jntadr[obj_body]])
    q[:, root_adr : root_adr + 3] = root_pos
    q[:, root_adr + 3 : root_adr + 7] = root_quat
    q[:, obj_adr : obj_adr + 3] = obj_pos
    q[:, obj_adr + 3 : obj_adr + 7] = obj_quat[:, [3, 0, 1, 2]]
    for j, name in enumerate(names):
        q[:, model.jnt_qposadr[match(model, "joint", name)]] = joint_pos[:, j]
    controls = q[:, model.jnt_qposadr[model.actuator_trnid[:, 0]]]
    weights = np.full(model.nq, 1.0)
    weights[root_adr : root_adr + 3] = [40, 40, 100]
    weights[obj_adr : obj_adr + 3] = 100
    for adr in [root_adr, obj_adr]:
        weights[adr + 3 : adr + 7] = 0
    # Initial velocity is derived consistently from reference qpos, including
    # local free-joint angular velocities; it is never reinserted afterwards.
    velocity = np.zeros(model.nv)
    mujoco.mj_differentiatePos(model, velocity, 1 / motion.fps, q[0], q[1])
    np.savez(out / "reference.npz", qpos=q, controls=controls, initial_qvel=velocity, fps=motion.fps)
    temporary.cleanup()
    return model, q, velocity, controls, weights, [root_adr + 3, obj_adr + 3], obj_body, names


def audit(model, q, v, target, obj_body, controls):
    import mujoco

    data = mujoco.MjData(model)
    obj_adr = int(model.jnt_qposadr[model.body_jntadr[obj_body]])
    feet = [match(model, "body", f"{s}_ankle_roll_link") for s in ["left", "right"]]
    penetration, slip, forces, feet_pos = [], [], [], []
    for frame, (qi, vi) in enumerate(zip(q, v, strict=True)):
        data.qpos[:] = qi
        data.qvel[:] = vi
        data.ctrl[:] = controls[min(max(frame - 1, 0), len(controls) - 1)]
        mujoco.mj_forward(model, data)
        feet_pos.append(data.xpos[feet].copy())
        depth = 0.0
        force = 0.0
        for c in range(data.ncon):
            contact = data.contact[c]
            depth = max(depth, -float(contact.dist))
            a, b = model.geom_bodyid[contact.geom]
            if obj_body in (a, b) and (a if b == obj_body else b) > 0:
                wrench = np.zeros(6)
                mujoco.mj_contactForce(model, data, c, wrench)
                force += float(abs(wrench[0]))
        penetration.append(depth)
        forces.append(force)
    feet_pos = np.array(feet_pos)
    # Foot-origin speed near the ground is a diagnostic, not a stance label.
    speed = np.linalg.norm(np.diff(feet_pos, axis=0)[:, :, :2], axis=2) * 50
    mask = feet_pos[:-1, :, 2] < 0.12
    slip = speed[mask]
    error = np.linalg.norm(q[:, obj_adr : obj_adr + 3] - target[: len(q), obj_adr : obj_adr + 3], axis=1)
    return {
        "object_rmse_m": float(np.sqrt(np.mean(error**2))),
        "object_max_error_m": float(error.max()),
        "root_min_z_m": float(q[:, 2].min()),
        "max_penetration_m": float(max(penetration)),
        "near_ground_foot_speed_p95_m_s": float(np.percentile(slip, 95)) if len(slip) else None,
        "object_contact_normal_force_max_N": float(max(forces)),
        "contact_measurement": "CPU forward reconstruction at executed states; includes robot-object contacts",
        "accepted": False,
        "training_data_ready": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=550)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--advance", type=int, default=5)
    parser.add_argument("--sigma", type=float, default=0.12)
    parser.add_argument("--arm-noise-scale", type=float, default=1.0)
    parser.add_argument("--leg-noise-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmstart", type=Path, help="optional seed_dynamics.py output")
    parser.add_argument(
        "--bfm-warmstart",
        choices=("full", "reference-arms"),
        help="automatically generate a frozen-BFM seed in this run",
    )
    parser.add_argument("--render-review", action="store_true")
    parser.add_argument(
        "--project-initial", action="store_true", help="repair the observed initial left-hand/table overlap"
    )
    args = parser.parse_args()
    if args.warmstart is not None and args.bfm_warmstart is not None:
        parser.error("--warmstart and --bfm-warmstart are mutually exclusive")
    cfg = Config(
        samples=args.samples,
        iterations=args.iterations,
        horizon=args.horizon,
        advance=args.advance,
        sigma=args.sigma,
        seed=args.seed,
    )
    out = run_dir("sequence_1019/09_lined_rect_basket_dynamics")
    print(f"OUTPUT={out}", flush=True)
    candidate = args.candidate.resolve()
    manifest = {
        "candidate": str(candidate),
        "algorithm": "annealed MPPI baseline; no virtual contact guidance",
        "config": asdict(cfg),
        "source_hashes": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                candidate / "motion.npz",
                candidate / "object_estimate.npz",
                Path(__file__).with_name("scene_stage4.py"),
            ]
        },
        "accepted": False,
        "training_data_ready": False,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    source_snapshot = out / "source"
    source_snapshot.mkdir()
    shutil.copy2(__file__, source_snapshot / Path(__file__).name)
    from robosimstudio.humanoid_real2sim import refinement

    for source in Path(refinement.__file__).parent.glob("*.py"):
        shutil.copy2(source, source_snapshot / source.name)
    for name in ("scene_stage4.py", "seed_dynamics.py", "review_dynamics.py"):
        shutil.copy2(Path(__file__).with_name(name), source_snapshot / name)
    model, target, velocity, controls, weights, quats, obj_body, _names = prepare(candidate, out)
    initial = target[0].copy()
    if args.project_initial:
        from robosimstudio.humanoid_real2sim.refinement.initial_state import project_initial

        selected = [
            match(model, "joint", f"left_{name}_joint")
            for name in [
                "shoulder_pitch",
                "shoulder_roll",
                "shoulder_yaw",
                "elbow",
                "wrist_roll",
                "wrist_pitch",
                "wrist_yaw",
            ]
        ]
        initial, projection = project_initial(model, initial, selected)
        (out / "initial_projection.json").write_text(json.dumps(projection, indent=2))
    np.savez(out / "initial_state.npz", qpos=initial, qvel=velocity)
    count = min(args.frames, len(target))
    if count < 2:
        raise ValueError("at least two frames required")
    target = target[:count]
    controls = controls[: count - 1]
    if args.bfm_warmstart is not None:
        command = [sys.executable, str(Path(__file__).with_name("seed_dynamics.py")), str(out), "--frames", str(count)]
        if args.bfm_warmstart == "reference-arms":
            command.append("--reference-arms")
        subprocess.run(command, check=True)
        args.warmstart = out / (
            "tracker_seed_reference_arms.npz" if args.bfm_warmstart == "reference-arms" else "tracker_seed.npz"
        )
    body_weights = np.zeros((model.nbody, 2))
    for name, pw, rw in [
        ("torso_link", 30.0, 10.0),
        ("left_ankle_roll_link", 30.0, 3.0),
        ("right_ankle_roll_link", 30.0, 3.0),
        ("left_wrist_yaw_link", 10.0, 2.0),
        ("right_wrist_yaw_link", 10.0, 2.0),
    ]:
        body_weights[match(model, "body", name)] = [pw, rw]
    wrists = [match(model, "body", f"{side}_wrist_yaw_link") for side in ("left", "right")]
    geom_kind = np.zeros(model.ngeom, dtype=np.int32)
    for g, b in enumerate(model.geom_bodyid):
        name = model.body(int(b)).name
        if b == obj_body:
            geom_kind[g] = 1
        elif b != 0 and not any(part in name for part in ("_elbow_", "_wrist_", "_hand_")):
            geom_kind[g] = 2
    backend = WarpBackend(
        model,
        initial,
        velocity,
        target,
        weights,
        quats,
        samples=cfg.samples,
        horizon=cfg.horizon,
        body_weights=body_weights,
        relative_body_ids=wrists,
        object_body=obj_body,
        contact_geom_kind=geom_kind,
    )
    manifest["objective_version"] = "body_foot_wrist_relative_contact_v2"
    manifest["body_objective_weights"] = body_weights.tolist()
    manifest["relative_wrist_weight"] = 100
    manifest["unwanted_object_contact_weight"] = 5
    backend.noise_scale = np.ones(model.nu)
    if (
        not np.isfinite([args.arm_noise_scale, args.leg_noise_scale]).all()
        or min(args.arm_noise_scale, args.leg_noise_scale) < 0
    ):
        raise ValueError("noise multipliers must be finite and nonnegative")
    for i, j in enumerate(model.actuator_trnid[:, 0]):
        name = model.joint(int(j)).name
        if any(
            name.endswith(f"{side}_{part}_joint")
            for side in ("left", "right")
            for part in (
                "shoulder_pitch",
                "shoulder_roll",
                "shoulder_yaw",
                "elbow",
                "wrist_roll",
                "wrist_pitch",
                "wrist_yaw",
            )
        ):
            backend.noise_scale[i] = args.arm_noise_scale
        elif any(
            name.endswith(f"{side}_{part}_joint")
            for side in ("left", "right")
            for part in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
        ):
            backend.noise_scale[i] = args.leg_noise_scale
    manifest["noise_scale"] = backend.noise_scale.tolist()
    manifest["project_initial"] = args.project_initial
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    start = time.perf_counter()
    for i in range(0, len(controls), cfg.advance):
        backend.commit(controls[i : i + cfg.advance])
    baseline_q, baseline_v = np.array(backend.history_q), np.array(backend.history_v)
    baseline = audit(model, baseline_q, baseline_v, target, obj_body, controls)
    baseline["wall_s"] = time.perf_counter() - start
    np.savez(out / "baseline.npz", qpos=baseline_q, qvel=baseline_v, controls=controls)
    (out / "baseline.json").write_text(json.dumps(baseline, indent=2))
    print("BASELINE", baseline, flush=True)
    backend.reset_execution(initial, velocity)

    def progress(record):
        record["root_z_m"] = float(backend.qpos[2])
        obj_adr = int(model.jnt_qposadr[model.body_jntadr[obj_body]])
        record["object_error_m"] = float(
            np.linalg.norm(backend.qpos[obj_adr : obj_adr + 3] - target[record["frame"], obj_adr : obj_adr + 3])
        )
        if record["frame"] % 50 == 0:
            np.savez(out / "checkpoint.npz", qpos=np.array(backend.history_q), qvel=np.array(backend.history_v))
        with (out / "progress.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)

    guess = controls
    if args.warmstart is not None:
        with np.load(args.warmstart, allow_pickle=False) as seed:
            guess = seed["controls"][: len(controls)].copy()
            if guess.shape != controls.shape or not np.isfinite(guess).all():
                raise ValueError("warmstart shape or values invalid")
            if not np.allclose(seed["qpos"][0], initial, atol=1e-7, rtol=0):
                raise ValueError("warmstart initial pose differs; use the same initialization options")
            expected = [model.joint(int(j)).name for j in model.actuator_trnid[:, 0]]
            if seed["actuator_joint_names"].tolist() != expected:
                raise ValueError("warmstart actuator order differs")
        manifest["warmstart"] = {
            "path": str(args.warmstart.resolve()),
            "sha256": hashlib.sha256(args.warmstart.read_bytes()).hexdigest(),
        }
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
        guess = np.clip(guess, backend.lower, backend.upper)
    optimized, records = optimize(backend, guess, cfg, progress)
    q, v = np.array(backend.history_q), np.array(backend.history_v)
    np.savez(
        out / "optimized.npz",
        qpos=q,
        qvel=v,
        controls=optimized,
        fps=50,
        actuator_joint_names=np.array([model.joint(int(j)).name for j in model.actuator_trnid[:, 0]]),
    )
    order = json.loads(DEFAULT_ORDER.read_text())
    export_motion(model, q, v, order["robot_joint_names"], order["robot_body_names"], out / "candidate_motion.npz")
    obj_adr = int(model.jnt_qposadr[model.body_jntadr[obj_body]])
    with np.load(candidate / "object_estimate.npz") as sidecar:
        source_time = sidecar["source_time_s"][:count] if "source_time_s" in sidecar else 98.0 + np.arange(count) / 50
    np.savez(
        out / "object_executed.npz",
        position_w=q[:, obj_adr : obj_adr + 3],
        quaternion_xyzw=q[:, obj_adr + 3 : obj_adr + 7][:, [1, 2, 3, 0]],
        fps=50,
        source_time_s=source_time,
        reference_frame=np.arange(count),
    )
    metrics = audit(model, q, v, target, obj_body, optimized)
    metrics["optimization_wall_s"] = records[-1]["wall_s"]
    # Independent uninterrupted CPU replay of saved controls, without optimizer state.
    backend.reset_execution(initial, velocity)
    for i in range(0, len(optimized), cfg.advance):
        backend.commit(optimized[i : i + cfg.advance])
    metrics["control_replay_max_qpos_error"] = float(np.max(np.abs(np.array(backend.history_q) - q)))
    metrics["baseline"] = baseline
    metrics["sampling_overflow_bits"] = backend.overflow_bits
    metrics["execution_backend"] = "MuJoCo CPU, identical RSS-compiled model; proposals use MuJoCo Warp"
    metrics["source_frames"] = count
    (out / "review.json").write_text(json.dumps(metrics, indent=2))
    (out / "stage_ledger.json").write_text(
        json.dumps(
            {
                "stage": "4-offline-refinement-experimental",
                "status": "computed_pending_review",
                "source_candidate": str(candidate),
                "accepted": False,
                "training_data_ready": False,
                "model_sha256": hashlib.sha256((out / "scene.mjb").read_bytes()).hexdigest(),
                "review": "review.json",
                "motion": "candidate_motion.npz",
                "object_track": "object_executed.npz",
            },
            indent=2,
        )
    )
    print("REVIEW", json.dumps(metrics), flush=True)
    if args.render_review:
        subprocess.run(
            [sys.executable, str(Path(__file__).with_name("review_dynamics.py")), str(out)],
            env={**os.environ, "MUJOCO_GL": "egl"},
            check=True,
        )


if __name__ == "__main__":
    main()
