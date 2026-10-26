"""Stage a kinematic brace/wipes package with direct original ScaleTrack USD references."""


def main():
    import argparse
    import hashlib
    import json
    import shutil
    import sys
    from pathlib import Path

    import numpy as np
    import yaml
    from scipy.spatial.transform import Rotation

    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    args = ap.parse_args()
    r = args.run
    root = Path("/home/galbot/workspace/ScaleTrack")
    name = "brace-dining-table-wipes-kinematic-v1"
    rel = "data/1019test/" + name
    out = r / "archive" / name
    out.mkdir(parents=True, exist_ok=False)
    meta = out / "metadata"
    (meta / "object_trajectories").mkdir(parents=True)
    p = r / "retarget/v1"
    motion = dict(np.load(p / "motion.npz"))
    n = len(motion["joint_pos"])
    ts = np.arange(n) / 50
    names = ["grey_armchair", "wet_wipes"]
    xyz, quat, lin, ang = [], [], [], []
    for obj, filename in zip(names, ["chair_track.npz", "wet_wipes.npz"], strict=True):
        z = np.load(p / filename)
        pos = z["position_w"]
        q = z["quaternion_xyzw"][:, [3, 0, 1, 2]]
        lv = np.gradient(pos, 0.02, axis=0)
        rot = Rotation.from_quat(z["quaternion_xyzw"])
        av = (rot[1:] * rot[:-1].inv()).as_rotvec() * 50
        av = np.vstack([av, av[-1]])
        source = "user_placed_static_wipes" if obj == "wet_wipes" else "inherited_kinematic_chair_estimate"
        np.savez_compressed(
            meta / "object_trajectories" / (obj + ".npz"),
            object_name=obj,
            fps=50.0,
            timestamps=ts,
            recording_time_s=83.5 + ts,
            position_w=pos,
            quaternion_wxyz=q,
            linear_velocity_w=lv,
            angular_velocity_w=av,
            measured_gt_valid=np.zeros(n, bool),
            kinematic_pose_defined=np.ones(n, bool),
            trajectory_source=source,
        )
        xyz.append(pos)
        quat.append(q)
        lin.append(lv)
        ang.append(av)
    motion.update(
        object_names=np.array(names),
        object_pos_w=np.stack(xyz, 1),
        object_quat_w=np.stack(quat, 1),
        object_lin_vel_w=np.stack(lin, 1),
        object_ang_vel_w=np.stack(ang, 1),
        timestamps=ts,
        recording_time_s=83.5 + ts,
        source_motion_frame=np.arange(n),
        interacting_object_names=np.array(["wet_wipes"]),
        object_trajectory_measured=np.asarray(False),
        object_trajectory_source=np.asarray("estimated_kinematic_reference_no_physics"),
    )
    np.savez_compressed(out / "motion.npz", **motion)
    original = root / "data/1019test/brace-dining-table-kinematic-v1"
    for filename in [
        "human.npz",
        "human_motion_world.npz",
        "source_sync.npz",
        "motion_order.json",
        "chair_track.npz",
        "wet_wipes.npz",
        "tracks.json",
    ]:
        shutil.copy2(p / filename, meta / filename)
    shutil.copy2(original / "metadata/camera_calibration.json", meta / "camera_calibration.json")
    shutil.copytree(r / "review", meta / "review")
    shutil.copy2(r / "scene/config.json", meta / "scene_review_config.json")
    config = next(iter(yaml.safe_load((original / "motions.yaml").read_text()).values()))
    config.update(
        path=rel + "/motion.npz",
        archive_status="user_requested_kinematic_archive",
        kinematic_reference_accepted_by_user=True,
        visual_gt="not_generated",
        visual_dataset_status="not_generated_for_wipes",
        dynamics="not_run",
        accepted=False,
        training_data_ready=False,
    )
    ss = config["scene_setup"]
    ss.update(
        status="kinematic_geometry_checked_original_assets_referenced",
        runtime_ready=False,
        runtime_note="No physics, support force or training validation; visual GT not generated.",
    )
    ss["camera"]["calibration_file"] = rel + "/metadata/camera_calibration.json"
    chair = dict(
        next(
            x
            for x in next(
                iter(yaml.safe_load((root / "data/1012test/1012-1/stage7-visual-gt/motions.yaml").read_text()).values())
            )["scene_setup"]["objects"]
            if x["id"] == "grey_armchair"
        )
    )
    chair["initial_position_m"] = xyz[0][0].tolist()
    chair["initial_orientation_wxyz"] = quat[0][0].tolist()
    wipes = {
        "id": "wet_wipes",
        "type": "wet_wipes",
        "asset_kind": "rigid_object",
        "scene_field": "wet_wipes",
        "usd_path": "source/scaletrack/scaletrack/assets/objects/1012/wet_wipes_v3.usd",
        "blend_path": "source/scaletrack/scaletrack/assets/objects/1012/wet_wipes_v3.blend",
        "scale": [1.0, 1.0, 1.0],
        "initial_position_m": xyz[1][0].tolist(),
        "initial_orientation_wxyz": quat[1][0].tolist(),
        "canonical_frame": "native_asset_default_prim",
        "trajectory_kind": "user_placed_static_kinematic_reference",
        "physics_validation": "not_run",
    }
    ss["object"] = wipes
    ss["objects"] = [chair, wipes]
    config["object_trajectories"] = {
        obj: {
            "path": rel + "/metadata/object_trajectories/" + obj + ".npz",
            "npz_object_index": i,
            "interacts_in_this_clip": obj == "wet_wipes",
            "quaternion_order": "wxyz",
            "coordinate_frame": "IKEA_1012_world",
            "measured": False,
        }
        for i, obj in enumerate(names)
    }
    config["interaction_events"].append(
        {
            "effector": "left_closed_fist",
            "surface": "wet_wipes",
            "source_recording_time_s": [86.5, 87.6],
            "clip_time_s": [3.0, 4.1],
            "status": "kinematic_geometry_only_not_force_measurement",
        }
    )
    (out / "motions.yaml").write_text(
        yaml.safe_dump({"hoi1019_brace_dining_table_wipes_kinematic_v1": config}, sort_keys=False)
    )
    sys.path.insert(0, str(root / "retarget3/src"))
    import retarget3.runtime  # noqa: F401
    from pxr import Usd, UsdGeom, UsdUtils

    assets = {}
    for asset_path in [ss["scene"]["usd_path"], chair["usd_path"], wipes["usd_path"], ss["camera"]["robot_usd"]]:
        asset = root / asset_path
        stage = Usd.Stage.Open(str(asset))
        assert stage and stage.GetDefaultPrim(), asset
        layers, dependencies, unresolved = UsdUtils.ComputeAllDependencies(str(asset))
        assets[asset_path] = {
            "sha256": hashlib.sha256(asset.read_bytes()).hexdigest(),
            "default_prim": str(stage.GetDefaultPrim().GetPath()),
            "meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage),
            "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
            "unresolved": list(map(str, unresolved)),
            "dependency_count": len(layers) + len(dependencies),
        }
        assets[asset_path]["runtime_builtin_dependencies"] = [str(x) for x in unresolved if str(x) == "OmniPBR.mdl"]
        assert all(str(x) == "OmniPBR.mdl" for x in unresolved), assets[asset_path]
    (meta / "asset_references.json").write_text(json.dumps(assets, indent=2))
    (meta / "provenance.json").write_text(
        json.dumps(
            {
                "run": str(r),
                "version": "v1",
                "original_asset_policy": "Direct initial assets references; no USD exported or copied",
                "motion_sha256": hashlib.sha256((out / "motion.npz").read_bytes()).hexdigest(),
                "review_proxy": "Wipes visual triangles extracted from original USD; RSS convex-hull collision proxy not physical validation of USD collider",
                "room_chair": "Inherited RSS room/chair visual proxies; initial USD physics not validated",
            },
            indent=2,
        )
    )
    (out / "README.md").write_text(
        "# 右手撑桌、左拳按湿巾\n\n83.5-91秒,375帧,50Hz。保留已审阅v15机器人动作,湿巾包在前版基础上再转180°(最终yaw0°)。\n\nUSD直接引用ScaleTrack assets中的初始wet_wipes_v3.usd、grey_armchair_v2.usd及原房间/机器人资产;无新USD导出或复制。物体轨迹为运动学估计,非实测。动力学、压力和训练未验证,视觉GT未生成。原无湿巾撑桌包保留。\n"
    )
    print(out)


if __name__ == "__main__":
    main()
