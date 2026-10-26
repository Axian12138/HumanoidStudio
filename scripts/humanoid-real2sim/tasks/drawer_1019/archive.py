"""Archive drawer v2 using the same scene assets as the prior hat package."""

import argparse
import hashlib
import json
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml

ST = Path("/home/galbot/workspace/ScaleTrack")
REL = "data/1019test/drawer-push-kinematic-v1"
HATREL = "data/1019test/hat-to-stand-kinematic-v1"


def dump(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    import trimesh
    from scipy.spatial.transform import Rotation

    sys.path.insert(0, str(ST / "retarget3/src"))
    import retarget3.runtime  # noqa: F401

    # isort: split
    import mujoco
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    a = ap.parse_args()
    run = a.run.resolve()
    src = run / "retarget/v2"
    out = ST / REL
    hat = ST / HATREL
    if out.exists():
        raise FileExistsError(out)
    out.mkdir()
    for folder in ["assets", "scene"]:
        shutil.copytree(hat / folder, out / folder)
    for p in (out / "scene").glob("*"):
        if p.suffix in [".xml", ".json", ".py"]:
            p.write_text(p.read_text().replace(str(hat), str(out)))
    config = json.loads((out / "scene/config.json").read_text())
    current = json.loads((run / "scene/config.json").read_text())
    ma = mujoco.MjModel.from_xml_path(config["robot_and_early_props"])
    mb = mujoco.MjModel.from_xml_path(current["robot_and_early_props"])
    for key in ["mesh_vert", "mesh_face", "body_pos", "body_quat", "geom_pos", "geom_quat", "geom_size", "jnt_axis"]:
        assert np.array_equal(getattr(ma, key), getattr(mb, key)), key
    meta = out / "metadata"
    meta.mkdir()
    for name in ["human_motion_world.npz", "source_sync.npz", "motion_order.json"]:
        shutil.copy2(src / name, meta / name)
    for folder in ["body_prior", "baseline"]:
        shutil.copytree(run / "retarget" / folder, meta / folder)
    shutil.copytree(src / "scene_tracks", meta / "scene_tracks")
    shutil.copytree(run / "review", meta / "review")
    for name in ["review.json", "native_drawer_hand_100hz.json"]:
        shutil.copy2(src / name, meta / "review" / name)
    for name in ["stage_ledger.json", "provenance.json", "README.md"]:
        shutil.copy2(run / name, meta / ("source_" + name))
    shutil.copy2(__file__, meta / "archive.py")
    z = dict(np.load(src / "motion.npz"))
    n = len(z["joint_pos"])
    fps = float(z["fps"])
    assert (n, fps) == (301, 50.0)
    times = np.arange(n) / fps
    manifest = json.loads((src / "scene_tracks/tracks.json").read_text())
    names = []
    pos = []
    quat = []
    lin = []
    ang = []
    (meta / "object_trajectories").mkdir()
    for item in manifest["objects"]:
        name = item["name"].removesuffix("_free")
        track = np.load(src / "scene_tracks" / item["path"])
        p = track["position_w"]
        q = track["quaternion_xyzw"][:, [3, 0, 1, 2]]
        assert p.shape == (n, 3) and q.shape == (n, 4)
        v = np.gradient(p, 1 / fps, axis=0)
        rot = Rotation.from_quat(q[:, [1, 2, 3, 0]])
        w = (rot[1:] * rot[:-1].inv()).as_rotvec() * fps
        w = np.vstack([w, w[-1]])
        names.append(name)
        pos.append(p)
        quat.append(q)
        lin.append(v)
        ang.append(w)
        np.savez_compressed(
            meta / "object_trajectories" / f"{name}.npz",
            position_w=p,
            quaternion_wxyz=q,
            linear_velocity_w=v,
            angular_velocity_w=w,
            fps=fps,
            timestamps=times,
            recording_time_s=49.5 + times,
            measured_gt_valid=np.zeros(n, bool),
        )
    drawer = np.load(src / "scene_tracks/drawer_slide.npz")["position"]
    z.update(
        object_names=np.asarray(names),
        object_pos_w=np.stack(pos, 1),
        object_quat_w=np.stack(quat, 1),
        object_lin_vel_w=np.stack(lin, 1),
        object_ang_vel_w=np.stack(ang, 1),
        timestamps=times,
        recording_time_s=49.5 + times,
        articulation_joint_names=np.asarray(["drawer_slide"]),
        articulation_joint_pos=drawer[:, None],
        articulation_joint_vel=np.gradient(drawer, 1 / fps)[:, None],
        object_trajectory_measured=np.asarray(False),
    )
    np.savez_compressed(out / "motion.npz", **z)
    # Remove the seven static drawer meshes from the reused room. Export the drawer
    # as an actual prismatic articulation, never a prescribed free rigid body.
    room = out / "assets/room_registered_with_coat_rack_and_drawer.usd"
    stage = Usd.Stage.Open(str(room))
    children = list(stage.GetDefaultPrim().GetChildren())
    assert len(children) == 317
    for prim in children[-7:]:
        stage.RemovePrim(prim.GetPath())
    room_new = out / "assets/room_registered_without_drawer.usd"
    stage.GetRootLayer().Export(str(room_new))
    del stage
    room.unlink()
    tree = ET.parse(current["robot_and_early_props"])
    body = tree.find("./worldbody/body[@name='drawer']")
    meshes = {e.get("name"): e for e in tree.findall("./asset/mesh")}
    meshdir = Path(tree.find("compiler").get("meshdir"))
    usd = out / "assets/drawer_articulation.usd"
    stage = Usd.Stage.CreateNew(str(usd))
    root = UsdGeom.Xform.Define(stage, "/DrawerAsset").GetPrim()
    stage.SetDefaultPrim(root)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.ArticulationRootAPI.Apply(root)
    position = np.fromstring(body.get("pos"), sep=" ")
    qbody = np.fromstring(body.get("quat"), sep=" ")

    def transform(x, pos, quat):
        x.AddTranslateOp().Set(Gf.Vec3d(*map(float, pos)))
        x.AddOrientOp().Set(Gf.Quatf(float(quat[0]), Gf.Vec3f(*map(float, quat[1:]))))

    for name in ["Base", "Drawer"]:
        x = UsdGeom.Xform.Define(stage, "/DrawerAsset/" + name)
        transform(x, position, qbody)
        UsdPhysics.RigidBodyAPI.Apply(x.GetPrim())
    fixed = UsdPhysics.FixedJoint.Define(stage, "/DrawerAsset/FixedBase")
    fixed.CreateBody1Rel().SetTargets(["/DrawerAsset/Base"])
    fixed.CreateLocalPos0Attr(Gf.Vec3f(*map(float, position)))
    fixed.CreateLocalRot0Attr(Gf.Quatf(float(qbody[0]), Gf.Vec3f(*map(float, qbody[1:]))))
    joint = UsdPhysics.PrismaticJoint.Define(stage, "/DrawerAsset/drawer_slide")
    joint.CreateBody0Rel().SetTargets(["/DrawerAsset/Base"])
    joint.CreateBody1Rel().SetTargets(["/DrawerAsset/Drawer"])
    joint.CreateAxisAttr("Y")
    joint.CreateLowerLimitAttr(0.0)
    joint.CreateUpperLimitAttr(0.32)
    for i, g in enumerate(body.findall("geom")):
        if g.get("type") == "mesh":
            spec = meshes[g.get("mesh")]
            path = Path(spec.get("file"))
            path = path if path.is_absolute() else meshdir / path
            mesh = trimesh.load(path, force="mesh", process=False)
            mesh.vertices *= np.fromstring(spec.get("scale", "1 1 1"), sep=" ")
        else:
            assert g.get("type") == "box"
            mesh = trimesh.creation.box(extents=2 * np.fromstring(g.get("size"), sep=" "))
        gp = np.fromstring(g.get("pos", "0 0 0"), sep=" ")
        gq = np.fromstring(g.get("quat", "1 0 0 0"), sep=" ")
        mesh.vertices = Rotation.from_quat(gq[[1, 2, 3, 0]]).apply(mesh.vertices) + gp
        u = UsdGeom.Mesh.Define(stage, f"/DrawerAsset/Drawer/mesh_{i}")
        u.CreatePointsAttr(mesh.vertices.astype(np.float32).tolist())
        u.CreateFaceVertexCountsAttr([3] * len(mesh.faces))
        u.CreateFaceVertexIndicesAttr(mesh.faces.ravel().tolist())
        u.CreateSubdivisionSchemeAttr("none")
        rgba = np.fromstring(g.get("rgba", ".7 .6 .4 1"), sep=" ")
        u.CreateDisplayColorAttr([Gf.Vec3f(*map(float, rgba[:3]))])
        if g.get("contype", "1") != "0":
            UsdPhysics.CollisionAPI.Apply(u.GetPrim())
            UsdPhysics.MeshCollisionAPI.Apply(u.GetPrim()).CreateApproximationAttr("convexHull")
        if rgba[-1] == 0 or g.get("group") == "3":
            u.CreateVisibilityAttr("invisible")
    stage.GetRootLayer().Save()
    room_stage = Usd.Stage.Open(str(room_new))
    mechanism = UsdGeom.Xform.Define(room_stage, "/Asset/DrawerMechanism").GetPrim()
    mechanism.GetReferences().AddReference("drawer_articulation.usd")
    room_stage.GetRootLayer().Save()
    template = yaml.safe_load((hat / "motions.yaml").read_text())
    cfg = next(iter(template.values()))
    cfg = json.loads(json.dumps(cfg).replace(HATREL, REL))
    cfg.update(
        source_frame_count=n,
        duration_seconds=n / fps,
        source_time_interval_seconds_inclusive=[49.5, 55.5],
        archive_status="existing_hoi_finetune_user_requested_archive",
        dynamics="not_run",
        training_data_ready=False,
        accepted=False,
    )
    scene = cfg["scene_setup"]
    scene["scene"]["usd_path"] = f"{REL}/assets/room_registered_without_drawer.usd"
    for item in scene["objects"]:
        idx = names.index(item["id"])
        item["initial_position_m"] = pos[idx][0].tolist()
        item["initial_orientation_wxyz"] = quat[idx][0].tolist()
    scene.pop("object", None)
    scene["scene"]["articulations"] = [
        {
            "id": "drawer",
            "relative_prim_path": "DrawerMechanism",
            "prim_path": "{ENV_REGEX_NS}/room/DrawerMechanism",
            "spawn": None,
            "scene_field": "drawer",
            "asset_kind": "articulation",
            "usd_path": f"{REL}/assets/drawer_articulation.usd",
            "initial_position_m": [0, 0, 0],
            "initial_orientation_wxyz": [1, 0, 0, 0],
            "initial_joint_position_m": {"drawer_slide": float(drawer[0])},
            "joint_limits_m": {"drawer_slide": [0, 0.32]},
            "actuators": {},
        }
    ]
    cfg["articulation_trajectories"] = {
        "drawer_slide": {
            "path": f"{REL}/metadata/scene_tracks/drawer_slide.npz",
            "field": "position",
            "unit": "m",
            "measured": False,
        }
    }
    schedule = {
        "time_base": "recording_time_s",
        "clip_start_s": 49.5,
        "contact_candidate_s": [51.1, 54.1],
        "source": "inherited source-constrained kinematic drawer trajectory; not measured force contact",
    }
    dump(meta / "contact_schedule.json", schedule)
    (out / "motions.yaml").write_text(yaml.safe_dump({"hoi1019_drawer_push_kinematic_v1": cfg}, sort_keys=False))
    original = np.load(src / "motion.npz")
    assert all(np.array_equal(z[k], original[k]) for k in original.files)
    assert np.array_equal(
        np.load(meta / "scene_tracks/drawer_slide.npz")["position"], z["articulation_joint_pos"][:, 0]
    )
    assert all(np.isfinite(v).all() for v in z.values() if v.dtype.kind in "fiu")
    for f in (out / "assets").glob("*.usd"):
        check = Usd.Stage.Open(str(f))
        assert check.GetDefaultPrim()
    dump(
        meta / "validation.json",
        {
            "frames": n,
            "fps": fps,
            "source_motion_fields_exact": True,
            "drawer_trajectory_exact": True,
            "scene_mjcf_geometry_exact": True,
            "usd_loads": True,
            "residual_actual_mesh_overlap_mm": 2.385,
            "dynamics": "not_run",
            "training_data_ready": False,
        },
    )
    (out / "README.md").write_text(
        "# Drawer push: existing HOI finetune v2\n\n301 frames, 50 Hz, source 49.5-55.5s inclusive. Preserves existing HOI and records a separate right-arm finetune; body prior and baseline are retained in metadata. Root, legs and left-hand berry motion remain unchanged.\n\nThe drawer is a prismatic articulation with its original 0-0.32m limits and synchronized scalar trajectory. RSS scene and all referenced mesh assets are included. USD and YAML separate the drawer from the static room.\n\nActual hand/drawer surfaces retain up to 2.385mm overlap; maximum sampled contact gap is 3.439mm. This is a user-requested kinematic archive, not a collision-free or physics/training acceptance. Dynamics and new visual GT were not run.\n"
    )
    dump(
        meta / "provenance.json",
        {
            "source_run": str(run),
            "source_version": "v2",
            "source_hoi": "scoped_candidate_v3 / 05_drawer_event_timing/v5",
            "source_motion_sha256": digest(src / "motion.npz"),
            "asset_base": str(hat),
            "user_requested_archive": True,
            "files_sha256": {str(p.relative_to(out)): digest(p) for p in sorted(out.rglob("*")) if p.is_file()},
        },
    )
    print(out)


if __name__ == "__main__":
    main()
