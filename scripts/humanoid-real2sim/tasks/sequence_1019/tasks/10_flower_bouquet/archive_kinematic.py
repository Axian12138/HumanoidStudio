# ruff: noqa: RUF001
"""Archive the reviewed flower motion while referencing initial public USD assets directly."""

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

ST = Path("/home/galbot/workspace/ScaleTrack")
ASSETS = ST / "source/scaletrack/scaletrack/assets"
NAME = "flower-to-basket-kinematic-v1"
REL = "data/1019test/" + NAME
KEY = "hoi1019_flower_to_basket_kinematic_v1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def validate(out):
    sys.path.insert(0, str(ST / "retarget3/src"))
    import retarget3.runtime  # noqa: F401
    from pxr import Usd, UsdGeom

    entry = yaml.safe_load((out / "motions.yaml").read_text())[KEY]
    z = np.load(out / "motion.npz", allow_pickle=False)
    n = len(z["joint_pos"])
    assert n == 1054 and z["joint_pos"].shape == (n, 43) and z["body_pos_w"].shape == (n, 50, 3)
    for k in [
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
        "object_pos_w",
        "object_quat_w",
        "object_lin_vel_w",
        "object_ang_vel_w",
    ]:
        assert np.isfinite(z[k]).all(), k
    assert np.allclose(np.linalg.norm(z["body_quat_w"], axis=-1), 1, atol=2e-5)
    assert np.allclose(np.linalg.norm(z["object_quat_w"], axis=-1), 1, atol=2e-5)
    assert np.allclose(z["timestamps"], np.arange(n) / 50)
    for i, name in enumerate(z["object_names"]):
        q = np.load(out / (str(name) + "_trajectory.npz"))
        for key, field in [
            ("position_w", "object_pos_w"),
            ("quaternion_wxyz", "object_quat_w"),
            ("linear_velocity_w", "object_lin_vel_w"),
            ("angular_velocity_w", "object_ang_vel_w"),
        ]:
            assert np.array_equal(q[key], z[field][:, i])
    sources = {}
    for item in [entry["scene_setup"]["scene"], *entry["scene_setup"]["objects"]]:
        p = (ST / item["usd_path"]).resolve()
        assert p.is_relative_to(ASSETS) and "1019_variants" not in p.parts
        stage = Usd.Stage.Open(str(p))
        assert stage and stage.GetDefaultPrim()
        assert UsdGeom.GetStageMetersPerUnit(stage) == 1 and UsdGeom.GetStageUpAxis(stage) == "Z"
        layers = [Path(x.realPath) for x in stage.GetUsedLayers() if x.realPath]
        assert all(x.exists() for x in layers)
        sources[str(p)] = {
            "sha256": sha(p),
            "default_prim": str(stage.GetDefaultPrim().GetPath()),
            "used_layers": {str(x): sha(x) for x in layers},
        }
    assert not list(out.rglob("*.usd")) and not list(out.rglob("*.usda")) and not list(out.rglob("*.usdc"))
    assert entry["scene_setup"]["object"] == entry["scene_setup"]["objects"][0]
    source = json.loads((out / "metadata/provenance.json").read_text())
    original = np.load(source["motion"])
    for key in ["joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w", "fps"]:
        assert np.array_equal(original[key], z[key]), key
    for path, expected in source["initial_asset_hashes"].items():
        assert sha(path) == expected
    result = {
        "status": "archive_structure_and_initial_usd_references_validated",
        "frames": n,
        "fps": 50,
        "body_fields_match_reviewed_motion_exactly": True,
        "sidecars_match_main_npz_exactly": True,
        "new_usd_files": 0,
        "public_assets_unchanged": True,
        "usd_sources": sources,
        "dynamics": "not_run",
        "visual_gt": "not_generated",
        "training_data_ready": False,
        "geometry_validation_scope": "Reviewed RSS proxies only; initial public USD geometry is not identical and requires separate interaction validation.",
    }
    dump(out / "metadata/validation.json", result)
    dump(
        out / "metadata/checksums.json",
        {str(f.relative_to(out)): sha(f) for f in sorted(out.rglob("*")) if f.is_file() and f.name != "checksums.json"},
    )
    print(json.dumps({k: v for k, v in result.items() if k != "usd_sources"}, indent=2))


def main():
    from scipy.spatial.transform import Rotation as R

    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--validate-only", action="store_true")
    a = p.parse_args()
    out = a.output.resolve()
    if a.validate_only:
        validate(out)
        return
    source = json.loads((a.run / "inputs/archive_source.json").read_text())
    run = Path(source["run"])
    ledger = json.loads((run / "stage_ledger.json").read_text())
    parent = json.loads((run / "inputs/parent.json").read_text())
    info = parent["parent"]
    motion = Path(ledger["motion"])
    out.mkdir(parents=True, exist_ok=False)
    meta = out / "metadata"
    meta.mkdir()
    z = dict(np.load(motion, allow_pickle=False))
    n = len(z["joint_pos"])
    fps = float(z["fps"])
    times = np.arange(n) / fps
    names = ["flower_bouquet", "small_wicker_basket"]
    ps = []
    qs = []
    vs = []
    ws = []
    entries = []
    trajectories = {}
    visual = json.loads((Path(info["assets"]) / "material_visual_provenance.json").read_text())
    for i, name in enumerate(names):
        tr = np.load(Path(ledger["tracks"]) / (name + ".npz"))
        pos = tr["position_w"]
        q = tr["quaternion_xyzw"]
        vel = np.gradient(pos, 1 / fps, axis=0)
        rr = R.from_quat(q)
        ang = (rr[1:] * rr[:-1].inv()).as_rotvec() * fps
        ang = np.vstack([ang, ang[-1]])
        qw = q[:, [3, 0, 1, 2]]
        ps.append(pos)
        qs.append(qw)
        vs.append(vel)
        ws.append(ang)
        np.savez_compressed(
            out / (name + "_trajectory.npz"),
            position_w=pos,
            quaternion_wxyz=qw,
            linear_velocity_w=vel,
            angular_velocity_w=ang,
            timestamps=times,
            recording_time_s=108 + times,
            fps=fps,
            measured_gt_valid=np.zeros(n, bool),
            trajectory_source="estimated_kinematic_reference",
        )
        trajectories[name] = {
            "path": REL + "/" + name + "_trajectory.npz",
            "npz_object_index": i,
            "coordinate_frame": "IKEA_1012_world",
            "quaternion_order": "wxyz",
            "measured": False,
        }
        entries.append(
            {
                "id": name,
                "type": name,
                "scene_field": name,
                "asset_kind": "rigid_object",
                "physics_mode": "dynamic",
                "usd_path": str((ASSETS / "objects/1012" / (name + "_v2.usd")).relative_to(ST)),
                "scale": visual[name]["scale"],
                "initial_position_m": pos[0].tolist(),
                "initial_orientation_wxyz": qw[0].tolist(),
                "trajectory_kind": "estimated_kinematic_reference",
                "physics_parameter_status": "Use existing public USD authored parameters; dynamics not validated",
            }
        )
    z.update(
        object_names=np.array(names),
        object_pos_w=np.stack(ps, 1),
        object_quat_w=np.stack(qs, 1),
        object_lin_vel_w=np.stack(vs, 1),
        object_ang_vel_w=np.stack(ws, 1),
        timestamps=times,
        recording_time_s=108 + times,
        source_motion_frame=np.arange(n),
        object_trajectory_measured=np.asarray(False),
        object_trajectory_source=np.asarray("estimated_kinematic_reference"),
    )
    np.savez_compressed(out / "motion.npz", **z)
    np.savez_compressed(
        meta / "source_frame_mapping.npz",
        source_frame=np.arange(n),
        recording_time_s=108 + times,
        output_time_s=times,
        phase=np.array(["kinematic_reference"] * n),
    )
    body = Path(info["historical_run"]) / "retarget/body_prior/10_flower_bouquet"
    for name in ["human_motion_world.npz", "human.npz", "source_sync.npz", "motion_order.json"]:
        shutil.copy2(body / name, meta / name)
    shutil.copytree(Path(ledger["tracks"]), meta / "scene_tracks")
    evidence = meta / "review_evidence"
    evidence.mkdir()
    current = run / "stage_ledger.json"
    chain = []
    while current.exists() and str(current) not in chain:
        chain.append(str(current))
        d = json.loads(current.read_text())
        dest = evidence / f"{len(chain):02d}_{current.parent.name}"
        dest.mkdir()
        shutil.copy2(current, dest / "stage_ledger.json")
        for f in (current.parent / "review").glob("*.json"):
            if f.stat().st_size < 3000000:
                shutil.copy2(f, dest / f.name)
        prior = d.get("parent_ledger") or d.get("previous_ledger")
        if not prior:
            break
        current = Path(prior)
    room = (
        ASSETS / "scenes/tianze_sim_data/IKEA_LivingRoom_Mesh_1012/IKEA_LivingRoom_Mesh_1012_drawer_renderable_v4.usd"
    )
    entry = {
        "path": REL + "/motion.npz",
        "weight": 1.0,
        "source_recording_id": "1-2026-10-19-16-31-26",
        "motion_fps": fps,
        "source_frame_count": n,
        "duration_seconds": n / fps,
        "slice_interval_seconds": [108, 108 + n / fps],
        "last_sample_recording_time_s": float(108 + times[-1]),
        "accepted": False,
        "kinematic_reference_accepted_by_user": True,
        "training_data_ready": False,
        "dynamics": "not_run",
        "visual_gt": "not_generated",
        "archive_status": "user_authorized_kinematic_archive_initial_usd_references",
        "trajectory_kind": "estimated_kinematic_reference",
        "scene_setup": {
            "version": 1,
            "status": "initial_public_assets_with_reviewed_motion_and_object_scales",
            "runtime_ready": False,
            "scene": {
                "usd_path": str(room.relative_to(ST)),
                "coordinate_frame": "IKEA_1012_world",
                "world_translation_m": [0, 0, 0],
                "world_orientation_wxyz": [1, 0, 0, 0],
                "require_no_embedded_object": True,
            },
            "object": entries[0],
            "objects": entries,
        },
        "object_trajectories": trajectories,
        "interaction_events": [
            {"event": event, "source_recording_time_s": t, "motion_time_s": t - 108}
            for event, t in [
                ("right_grasp_established", 113.2),
                ("left_deep_hook_pull_start", 118.7),
                ("left_pull_complete", 120.0),
                ("flower_over_rim_raise_start", 120.2),
                ("right_release_start", 123.2),
                ("basket_push_start", 125.0),
                ("basket_stopped", 127.3),
            ]
        ],
    }
    (out / "motions.yaml").write_text(yaml.safe_dump({KEY: entry}, sort_keys=False, allow_unicode=True))
    paths = [room, *[ST / x["usd_path"] for x in entries]]
    dump(
        meta / "provenance.json",
        {
            "motion": str(motion),
            "motion_sha256": sha(motion),
            "stage_ledger": str(run / "stage_ledger.json"),
            "review_chain": chain,
            "initial_asset_hashes": {str(x): sha(x) for x in paths},
            "asset_contract": "Direct references to pre-existing initial public USD. No USD copied, exported, authored or modified.",
            "scales_source": str(Path(info["assets"]) / "material_visual_provenance.json"),
            "reviewed_geometry": info["assets"],
            "geometry_difference": "RSS flower paper wrapper and collision proxies were edited; initial USD retains original geometry. Basket scale preserved; room uses public v4 rather than exported registered room. Prior proxy contact checks are not initial USD validation.",
            "dynamics": "not_run",
            "visual_gt": "not_generated",
        },
    )
    shutil.copy2(__file__, meta / "archive_kinematic.py")
    (out / "README.md").write_text(
        "# 花束放入篮子：运动学归档\n\n1054帧，50Hz，录制108.00–129.06秒。包含最后审阅的拿花、深勾拉篮、抬高越沿、放入释放、推进后停止与撤手。\n\n直接引用ScaleTrack公共初始USD，无新增或复制USD。尺度在motions.yaml中表达，源USD保持不变。当前RSS花束纸套、碰撞代理和房间局部重建与初始USD存在差异；已保留说明和来源，原代理的几何通过不等于初始USD交互验证。\n\n物体轨迹是运动学估计；动力学、初始USD下的接触/支撑和训练验收未运行，无新RGB/mask/DINO。\n"
    )
    validate(out)


if __name__ == "__main__":
    main()
