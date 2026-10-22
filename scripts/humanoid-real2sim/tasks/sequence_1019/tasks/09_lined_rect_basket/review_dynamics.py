"""Export and render an existing offline run; no optimization or input changes."""

# ruff: noqa: RUF001 -- Chinese review-page punctuation
from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path

import numpy as np

from robosimstudio.humanoid_real2sim.refinement.export import export_motion, unique_suffix
from robosimstudio.humanoid_real2sim.refinement.metrics import contact_diagnostics
from robosimstudio.humanoid_real2sim.refinement.render import render_comparison
from robosimstudio.humanoid_real2sim.track.launcher import DEFAULT_ORDER


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--skip-render", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco

    path = args.run.resolve()
    model = mujoco.MjModel.from_binary_path(str(path / "scene.mjb"))
    reference = np.load(path / "reference.npz")["qpos"]
    baseline = np.load(path / "baseline.npz")["qpos"]
    with np.load(path / "optimized.npz") as data:
        q, v = data["qpos"].copy(), data["qvel"].copy()
        controls = data["controls"].copy()
    order = json.loads(DEFAULT_ORDER.read_text())
    export_motion(model, q, v, order["robot_joint_names"], order["robot_body_names"], path / "candidate_motion.npz")
    obj = model.body("lined_rect_basket").id
    adr = int(model.jnt_qposadr[model.body_jntadr[obj]])
    manifest = json.loads((path / "manifest.json").read_text())
    with np.load(Path(manifest["candidate"]) / "object_estimate.npz") as source:
        times = source["source_time_s"][: len(q)] if "source_time_s" in source else 98 + np.arange(len(q)) / 50
    np.savez(
        path / "object_executed.npz",
        position_w=q[:, adr : adr + 3],
        quaternion_xyzw=q[:, adr + 3 : adr + 7][:, [1, 2, 3, 0]],
        fps=50,
        source_time_s=times,
        reference_frame=np.arange(len(q)),
    )
    report = json.loads((path / "review.json").read_text())
    feet = [unique_suffix(model, "body", f"{s}_ankle_roll_link") for s in ("left", "right")]
    contacts = contact_diagnostics(model, q, v, controls, obj, feet)
    np.savez(path / "contacts.npz", **contacts)
    report["object_contact_normal_force_max_N"] = float(
        (contacts["left_object_normal_force_N"] + contacts["right_object_normal_force_N"]).max()
    )
    with np.load(path / "baseline.npz") as base:
        bc = contact_diagnostics(model, base["qpos"], base["qvel"], base["controls"], obj, feet)
    report["baseline"]["object_contact_normal_force_max_N"] = float(
        (bc["left_object_normal_force_N"] + bc["right_object_normal_force_N"]).max()
    )
    stance_speed = contacts["loaded_foot_contact_speed_m_s"][contacts["loaded_foot_contact"]]
    stance_p95 = float(np.percentile(stance_speed, 95)) if len(stance_speed) else None
    report["contact_diagnostics"] = {
        "initial_penetration_m": float(contacts["penetration_m"][0]),
        "max_penetration_m": float(contacts["penetration_m"].max()),
        "loaded_foot_contact_speed_p95_m_s": stance_p95,
        "loaded_foot_contact_frames": int(contacts["loaded_foot_contact"].sum()),
        "left_contact_frames": int(np.sum(contacts["left_object_normal_force_N"] > 1)),
        "right_contact_frames": int(np.sum(contacts["right_object_normal_force_N"] > 1)),
        "bilateral_contact_frames": int(
            np.sum((contacts["left_object_normal_force_N"] > 1) & (contacts["right_object_normal_force_N"] > 1))
        ),
        "method": "forward dynamics at saved physical qpos/qvel and preceding control; 50 Hz, substep peaks not covered",
    }
    lift = reference[: len(q), adr + 2] > reference[0, adr + 2] + 0.03
    bilateral = (contacts["left_object_normal_force_N"] > 1) & (contacts["right_object_normal_force_N"] > 1)
    coverage = float(np.mean(bilateral[lift])) if lift.any() else 0.0
    dots = np.abs(np.sum(q[:, adr + 3 : adr + 7] * reference[: len(q), adr + 3 : adr + 7], axis=1))
    object_angles = np.rad2deg(2 * np.arccos(np.clip(dots, 0, 1)))
    torso = unique_suffix(model, "body", "torso_link")
    data = mujoco.MjData(model)
    torso_angles = []
    for actual, desired in zip(q, reference[: len(q)], strict=True):
        data.qpos[:] = actual
        mujoco.mj_forward(model, data)
        aq = data.xquat[torso].copy()
        data.qpos[:] = desired
        mujoco.mj_forward(model, data)
        torso_angles.append(np.rad2deg(2 * np.arccos(np.clip(abs(aq @ data.xquat[torso]), 0, 1))))
    report["quality_diagnostics"] = {
        "object_final_error_m": float(np.linalg.norm(q[-1, adr : adr + 3] - reference[len(q) - 1, adr : adr + 3])),
        "object_orientation_p95_deg": float(np.percentile(object_angles, 95)),
        "torso_orientation_p95_deg": float(np.percentile(torso_angles, 95)),
        "bilateral_loaded_contact_fraction_during_lift": coverage,
        "lift_mask_definition": "reference object origin > initial height + 0.03 m",
        "unintended_object_contact_frames": int((contacts["unintended_object_normal_force_N"] > 1).sum()),
        "unintended_object_contact_definition": "robot other than elbow/forearm/wrist/hand links; > 1 N",
    }
    report["automatic_gates"] = {
        "bilateral_contact_coverage_above_080": coverage >= 0.8,
        "object_orientation_p95_below_20deg": report["quality_diagnostics"]["object_orientation_p95_deg"] < 20,
        "torso_orientation_p95_below_20deg": report["quality_diagnostics"]["torso_orientation_p95_deg"] < 20,
        "no_unintended_robot_object_contact": report["quality_diagnostics"]["unintended_object_contact_frames"] == 0,
        "loaded_foot_contact_p95_below_010m_s": stance_p95 is not None and stance_p95 < 0.10,
        "root_above_045m": bool(q[:, 2].min() > 0.45),
        "object_max_error_below_010m": bool(
            np.max(np.linalg.norm(q[:, adr : adr + 3] - reference[: len(q), adr : adr + 3], axis=1)) < 0.10
        ),
        "penetration_below_001m": bool(contacts["penetration_m"].max() < 0.01),
        "control_replay_error_below_1e_minus_6": report["control_replay_max_qpos_error"] < 1e-6,
    }
    report["automatic_gates_passed"] = all(report["automatic_gates"].values())
    report["accepted"] = False
    report["training_data_ready"] = False
    (path / "review.json").write_text(json.dumps(report, indent=2))
    if not args.skip_render:
        render_comparison(model, reference, baseline, q, path)
    elif not (path / "comparison.mp4").is_file():
        raise ValueError("--skip-render requires an existing comparison.mp4")
    body = """<!doctype html><html lang="zh"><meta charset="utf-8"><title>Rect basket · 动力学 refinement 实验</title>
<style>body{background:#101721;color:#e7edf6;font:16px system-ui;margin:32px auto;max-width:1600px;padding:0 24px}
video{width:100%;background:#000}pre{white-space:pre-wrap;background:#192332;padding:24px;border-radius:12px}
a{color:#9fcbff}.notice{color:#ffd18c}</style>
<h1>Rect basket · 离线动力学 refinement</h1>
<p class="notice">实验候选，未通过训练数据验收。左：运动学参考；中：原关节目标的物理执行；右：优化控制的物理执行。</p>
<p>此页回放保存的轨迹。物理执行中篮子是自由体，无焊接、辅助力或逐帧设置物体位姿。机器人显示编译模型的碰撞几何；原 Stage4 页面保持不变。</p>
<video controls preload="metadata" src="comparison.mp4"></video>
<h2>实测结果</h2><pre>REPORT</pre>
<p><a href="review.json">完整指标</a> · <a href="manifest.json">输入与配置</a> · <a href="candidate_motion.npz">实测机器人 motion</a> · <a href="object_executed.npz">实测物体轨迹</a></p></html>"""
    (path / "index.html").write_text(
        body.replace("REPORT", html.escape(json.dumps(report, indent=2, ensure_ascii=False)))
    )
    print(path / "index.html")


if __name__ == "__main__":
    main()
