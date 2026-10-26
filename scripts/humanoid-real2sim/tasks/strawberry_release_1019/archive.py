"""Stage the accepted strawberry motion with the exact reviewed mesh geometry."""


def main():
    import argparse
    import hashlib
    import json
    import shutil
    import sys
    from pathlib import Path

    import numpy as np
    import trimesh
    import yaml
    from scipy.spatial.transform import Rotation

    sys.path.insert(0, "/home/galbot/workspace/ScaleTrack/retarget3/src")
    import retarget3.runtime  # noqa: F401
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    args = ap.parse_args()
    run = args.run.resolve()
    src = run / "retarget/v11"
    out = run / "archive/table-strawberry-to-basket-kinematic-v1"
    out.mkdir(parents=True, exist_ok=False)
    meta = out / "metadata"
    assets = out / "assets"
    meta.mkdir()
    assets.mkdir()
    rel = "data/1019test/table-strawberry-to-basket-kinematic-v1"
    st = Path("/home/galbot/workspace/ScaleTrack")
    original = Path("outputs/humanoid-real2sim/2026-10-20/sequence_1019/runs/21-08-52.042738+0800")
    input_assets = st / "data/1012test/hoi_stage2/scene/assets"
    geo = json.loads((src / "geometry_config.json").read_text())
    validation = {
        "source_version": "v11",
        "frames": 450,
        "fps": 50,
        "dynamics": "not_run",
        "training_data_ready": False,
        "mesh_roundtrip": {},
    }

    def mesh(stage, path, source, color, collision=False, hidden=False):
        t = trimesh.load(source, force="mesh", process=False)
        u = UsdGeom.Mesh.Define(stage, path)
        u.CreatePointsAttr(t.vertices.astype(np.float32).tolist())
        u.CreateFaceVertexCountsAttr([3] * len(t.faces))
        u.CreateFaceVertexIndicesAttr(t.faces.ravel().tolist())
        u.CreateSubdivisionSchemeAttr("none")
        u.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        if collision:
            UsdPhysics.CollisionAPI.Apply(u.GetPrim())
            UsdPhysics.MeshCollisionAPI.Apply(u.GetPrim()).CreateApproximationAttr(
                "convexHull" if collision == "convexHull" else "none"
            )
        if hidden:
            u.CreateVisibilityAttr("invisible")
        validation["mesh_roundtrip"][path] = {
            "source": str(Path(source).resolve()),
            "vertices": len(t.vertices),
            "faces": len(t.faces),
            "maximum_vertex_float32_error_m": float(np.max(abs(np.asarray(u.GetPointsAttr().Get()) - t.vertices))),
        }

    def stage_at(name):
        s = Usd.Stage.CreateNew(str(assets / name))
        root = UsdGeom.Xform.Define(s, "/Asset").GetPrim()
        s.SetDefaultPrim(root)
        UsdGeom.SetStageUpAxis(s, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(s, 1.0)
        return s, root

    s, root = stage_at("reviewed_room.usdc")
    mesh(s, "/Asset/Room", geo["room_world_mesh"], (0.5, 0.5, 0.5), True)
    chair = UsdGeom.Xform.Define(s, "/Asset/Chair")
    chair.AddTranslateOp().Set(Gf.Vec3d(-0.5523157716, -2.1457662582, 0.001))
    chair.AddOrientOp().Set(Gf.Quatf(0.9987785220, Gf.Vec3f(0, 0, -0.0494105928)))
    mesh(
        s,
        "/Asset/Chair/Visual",
        "outputs/humanoid-real2sim/2026-10-20/sequence_1019/curated_snapshot/tasks/07_armchair/grey_armchair_visual.obj",
        (0.5, 0.5, 0.5),
        True,
    )
    s.GetRootLayer().Save()
    for name, mass, friction, color in [
        ("table_strawberry", 0.06, 0.7, (0.8, 0.07, 0.12)),
        ("lined_rect_basket", 0.6, 0.7, (0.55, 0.45, 0.3)),
    ]:
        s, root = stage_at(name + ".usdc")
        UsdPhysics.RigidBodyAPI.Apply(root)
        UsdPhysics.MassAPI.Apply(root).CreateMassAttr(mass)
        mat = UsdPhysics.MaterialAPI.Apply(UsdGeom.Scope.Define(s, "/Asset/PhysicsMaterial").GetPrim())
        mat.CreateStaticFrictionAttr(friction)
        mat.CreateDynamicFrictionAttr(friction)
        from pxr import UsdShade

        material = UsdShade.Material.Define(s, "/Asset/PhysicsMaterial")
        UsdShade.MaterialBindingAPI.Apply(root).Bind(material, materialPurpose="physics")
        visual = (
            input_assets / "object_strawberry_existing_0_0.obj"
            if name == "table_strawberry"
            else original / "scene/08_table_strawberry/basket.obj"
        )
        mesh(s, "/Asset/Visual", visual, color, "convexHull" if name == "table_strawberry" else False)
        if name == "lined_rect_basket":
            for i in range(1, 6):
                mesh(
                    s,
                    f"/Asset/Collider{i}",
                    input_assets / f"object_lined_rect_basket_{i}_0.obj",
                    color,
                    "convexHull",
                    True,
                )
        s.GetRootLayer().Save()
    z = dict(np.load(src / "motion.npz"))
    names = ["table_strawberry", "lined_rect_basket"]
    ps = []
    qs = []
    vs = []
    ws = []
    for name in names:
        t = dict(np.load(src / (name + ".npz")))
        p = t["position_w"]
        q = t["quaternion_xyzw"]
        v = np.gradient(p, 0.02, axis=0)
        rot = Rotation.from_quat(q)
        w = (rot[1:] * rot[:-1].inv()).as_rotvec() / 0.02
        w = np.vstack([w, w[-1]])
        ps.append(p)
        qs.append(q[:, [3, 0, 1, 2]])
        vs.append(v)
        ws.append(w)
        np.savez_compressed(
            out / (name + "_trajectory.npz"),
            **t,
            quaternion_wxyz=qs[-1],
            linear_velocity_w=v,
            angular_velocity_w=w,
            measured_gt_valid=np.zeros(450, bool),
        )
    z.update(
        object_names=np.asarray(names),
        object_pos_w=np.stack(ps, 1),
        object_quat_w=np.stack(qs, 1),
        object_lin_vel_w=np.stack(vs, 1),
        object_ang_vel_w=np.stack(ws, 1),
        timestamps=np.arange(450) / 50,
        recording_time_s=90 + np.arange(450) / 50,
        object_trajectory_measured=np.asarray(False),
        object_trajectory_source=np.asarray("hybrid_kinematic_estimate"),
        interacting_object_names=np.asarray(names),
    )
    np.savez_compressed(out / "motion.npz", **z)
    for name in [
        "motion_order.json",
        "human.npz",
        "source_sync.npz",
        "geometry_config.json",
        "table_strawberry_successor_09.npz",
    ]:
        shutil.copy2(src / name, meta / name)
    shutil.copy2(original / "analysis/08_table_strawberry/human_motion_world.npz", meta / "human_motion_world.npz")
    for name in [
        "trajectory_fit.json",
        "track_update.json",
        "support_projection.json",
        "joint_audit.json",
        "dense_geometry_review.json",
        "object_sweep_review.json",
    ]:
        if (src / name).exists():
            shutil.copy2(src / name, meta / name)
    shutil.copy2(src / "body_audit/report.json", meta / "body_audit.json")
    shutil.copy2(run / "review/self_v11.json", meta / "self_audit.json")
    shutil.copy2(run / "review/player_validation_v11.json", meta / "player_validation.json")
    objects = []
    for i, name in enumerate(names):
        objects.append(
            {
                "id": name,
                "type": name,
                "asset_kind": "rigid_object",
                "scene_field": name,
                "usd_path": f"{rel}/assets/{name}.usdc",
                "scale": [1.0, 1.0, 1.0],
                "initial_position_m": ps[i][0].tolist(),
                "initial_orientation_wxyz": qs[i][0].tolist(),
                "trajectory_kind": "estimated_kinematic_reference",
                "physics_parameter_status": "RSS scene parameters copied; manipulation dynamics not verified",
            }
        )
    cfg = {
        "path": f"{rel}/motion.npz",
        "weight": 1.0,
        "source_recording_id": "1-2026-10-19-16-31-26",
        "motion_fps": 50.0,
        "source_frame_count": 450,
        "duration_seconds": 9.0,
        "slice_interval_seconds": [90.0, 99.0],
        "accepted": False,
        "kinematic_reference_accepted_by_user": True,
        "training_data_ready": False,
        "dynamics": "not_run",
        "visual_gt": "not_generated",
        "archive_status": "user_accepted_v11_kinematic",
        "trajectory_kind": "estimated_kinematic_reference",
        "scene_setup": {
            "version": 1,
            "status": "reviewed_RSS_geometry_kinematic_only",
            "runtime_ready": False,
            "scene": {
                "usd_path": f"{rel}/assets/reviewed_room.usdc",
                "coordinate_frame": "IKEA_1012_world",
                "world_translation_m": [0.0, 0.0, 0.0],
                "world_orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
                "require_no_embedded_object": True,
            },
            "object": objects[0],
            "objects": objects,
        },
        "object_trajectories": {n: {"path": f"{rel}/{n}_trajectory.npz", "measured": False} for n in names},
        "interaction_events": [
            {"event": "grasp", "source_recording_time_s": 92.4},
            {"event": "release", "source_recording_time_s": 97.0},
        ],
        "successor_connection": {
            "switch_source_time_s": 98.22,
            "retain_this_motion_through_s": 98.2,
            "recompute_velocities_after_composition": True,
            "strawberry_successor_track": f"{rel}/metadata/table_strawberry_successor_09.npz",
        },
    }
    (out / "motions.yaml").write_text(
        yaml.safe_dump({"hoi1019_table_strawberry_to_basket_kinematic_v1": cfg}, sort_keys=False)
    )
    for f in assets.glob("*.usdc"):
        s = Usd.Stage.Open(str(f))
        assert s.GetDefaultPrim()
    archive = np.load(out / "motion.npz")
    source = np.load(src / "motion.npz")
    assert all(np.array_equal(source[k], archive[k]) for k in source.files)
    for i, name in enumerate(names):
        t = np.load(out / (name + "_trajectory.npz"))
        assert np.array_equal(t["position_w"], archive["object_pos_w"][:, i])
        assert np.array_equal(t["quaternion_wxyz"], archive["object_quat_w"][:, i])
    assert all(np.isfinite(v).all() for v in z.values() if v.dtype.kind in "fiu")
    assert np.allclose(np.linalg.norm(z["body_quat_w"], axis=-1), 1, atol=1e-5)
    assert np.allclose(np.linalg.norm(z["object_quat_w"], axis=-1), 1, atol=1e-5)
    validation.update(
        source_motion_fields_exact=True, object_sidecars_exact=True, usd_reopened=True, quaternion_norms_valid=True
    )
    (meta / "validation.json").write_text(json.dumps(validation, indent=2))
    (out / "README.md").write_text(
        "# Table strawberry into basket: accepted v11\n\n450 frames at50Hz, recording90-98.98s. User accepted shoulder abduction (right roll≈-58°) and negative elbow (≈-11°) at97s, superseding rejected positive-elbow v8. Seven robot fields exactly match the reviewed v11. Strawberry and basket tracks are synchronized kinematic estimates; no RGB/DINO is included.\n\nAssets export the exact reviewed room/chair/object mesh geometry, with simple display colors. Basket collision is five separate hulls, not a hull over the opening. RSS masses/friction are copied; physics and training are not validated. Hidden RSS support proxy is not exported as a second tabletop collision; the actual room mesh provides the table.\n\nHand/basket minimum sampled clearance5.814mm; berry/basket0.2mm; room edge intersections0; no newly worsened self overlap>1mm. Modified window joint limits/speeds pass. Inherited successor finger speed violations and nonzero thumb-root trajectories remain and are not certified by this local arm acceptance. Postrelease is a kinematic support-projected estimate, not the original physical rest trace.\n\nComposition: overlaps preceding brace90-90.98s (thumb-root convention needs unification); use this robot path through98.20s and original basket path from98.22s. Recompute velocity fields on the composed timeline. The full successor strawberry track is included but the basket archive has not been changed.\n"
    )

    def digest(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()

    provenance = {
        "source_run": str(run),
        "source_version": "v11",
        "user_accepted": True,
        "source_motion_sha256": digest(src / "motion.npz"),
        "files_sha256": {str(p.relative_to(out)): digest(p) for p in sorted(out.rglob("*")) if p.is_file()},
    }
    (meta / "provenance.json").write_text(json.dumps(provenance, indent=2))
    print(out)


if __name__ == "__main__":
    main()
