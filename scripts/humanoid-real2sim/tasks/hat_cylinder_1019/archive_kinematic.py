# ruff: noqa: RUF001
"""Archive natural_v4 with the exact reviewed mesh geometry (ScaleTrack Python)."""

import argparse
import hashlib
import json
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[4]
ST = Path("/home/galbot/workspace/ScaleTrack")
REL = "data/1019test/hat-to-stand-kinematic-v1"


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()


def dump(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def main():
    import trimesh
    from scipy.spatial.transform import Rotation

    sys.path.insert(0, "/home/galbot/workspace/ScaleTrack/retarget3/src")
    import retarget3.runtime  # noqa: F401
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    run, out = a.run.resolve(), a.output.resolve()
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    src = run / "retarget/natural_v4"
    meta = out / "metadata"
    meta.mkdir()
    copied = {}

    def copy_asset(source):
        source = Path(source).resolve()
        if source in copied:
            return copied[source]
        target = out / "assets/source" / (hashlib.sha256(str(source).encode()).hexdigest()[:12] + "_" + source.name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied[source] = target
        if source.suffix.lower() == ".obj":
            lines = []
            for raw_line in source.read_text().splitlines():
                line = raw_line
                if line.startswith("mtllib "):
                    line = "mtllib " + copy_asset(source.parent / line[7:].strip()).name
                lines.append(line)
            target.write_text("\n".join(lines) + "\n")
        elif source.suffix.lower() == ".mtl":
            lines = []
            for raw_line in source.read_text().splitlines():
                line = raw_line
                if line.startswith(("map_", "bump ")):
                    key, value = line.split(maxsplit=1)
                    line = key + " " + copy_asset(source.parent / value).name
                lines.append(line)
            target.write_text("\n".join(lines) + "\n")
        return target

    config = json.loads((run / "scene/config.json").read_text())
    tree = ET.parse(config["robot_and_early_props"])
    xmlroot = tree.getroot()
    compiler = xmlroot.find("compiler")
    meshdir = Path(compiler.get("meshdir", str(run / "scene")))
    meshes = {}
    for m in xmlroot.findall("./asset/mesh"):
        path = Path(m.get("file"))
        if not path.is_absolute():
            path = meshdir / path
        meshes[m.get("name")] = (path, np.fromstring(m.get("scale", "1 1 1"), sep=" "))
        m.set("file", str(copy_asset(path)))
    compiler.attrib.pop("meshdir", None)
    for tex in xmlroot.findall("./asset/texture"):
        if tex.get("file"):
            tex.set("file", str(copy_asset(tex.get("file"))))
    (out / "scene").mkdir()
    xmlpath = out / "scene/robot_props.xml"
    tree.write(xmlpath, encoding="unicode")
    config["robot_and_early_props"] = str(xmlpath)
    config["room_mesh"] = str(copy_asset(config["room_mesh"]))
    for item in config.get("additional_static_assets", []):
        item["path"] = str(copy_asset(item["path"]))
    for item in config["additional_bodies"]:
        item["visual_mesh"] = str(copy_asset(item["visual_mesh"]))
        item["collision_meshes"] = [str(copy_asset(p)) for p in item.get("collision_meshes", [])]
        if not item["collision_meshes"]:
            item.pop("collision_meshes")
    dump(out / "scene/config.json", config)
    shutil.copy2(ROOT / "scripts/humanoid-real2sim/tasks/sequence_1019/unified_scene.py", out / "scene/rss_scene.py")
    for name in ["human_motion_world.npz", "source_sync.npz", "motion_order.json"]:
        shutil.copy2(src / name, meta / name)
    shutil.copytree(src / "scene_tracks", meta / "scene_tracks")
    shutil.copytree(run / "review", meta / "review")
    for name in ["README.md", "stage_ledger.json"]:
        shutil.copy2(run / name, meta / ("source_" + name))
    shutil.copy2(run / "scene/provenance.json", meta / "source_geometry.json")
    shutil.copy2(src / "natural_carry_review.json", meta / "natural_carry_review.json")
    shutil.copy2(__file__, meta / "archive_kinematic.py")
    shutil.copytree(run / "retarget/human_body_prior", meta / "human_body_prior")
    motion = dict(np.load(src / "motion.npz", allow_pickle=False))
    n, fps = len(motion["joint_pos"]), float(motion["fps"])
    assert n == 751 and fps == 50
    times = np.arange(n) / fps
    tracks = json.loads((src / "scene_tracks/tracks.json").read_text())
    names, positions, quats, linears, angulars = [], [], [], [], []
    trajectories = {}
    for item in tracks["objects"]:
        name = item["name"].removesuffix("_free")
        z = np.load(src / "scene_tracks" / item["path"])
        p, q = z["position_w"], z["quaternion_xyzw"]
        assert p.shape == (n, 3) and q.shape == (n, 4)
        v = np.gradient(p, 1 / fps, axis=0)
        rotations = Rotation.from_quat(q)
        # World angular velocity uses R_next * R_current^-1.
        w = (rotations[1:] * rotations[:-1].inv()).as_rotvec() * fps
        w = np.vstack([w, w[-1]])
        qw = q[:, [3, 0, 1, 2]]
        names.append(name)
        positions.append(p)
        quats.append(qw)
        linears.append(v)
        angulars.append(w)
        file = meta / "object_trajectories" / (name + ".npz")
        file.parent.mkdir(exist_ok=True)
        np.savez_compressed(
            file,
            position_w=p,
            quaternion_wxyz=qw,
            linear_velocity_w=v,
            angular_velocity_w=w,
            fps=fps,
            timestamps=times,
            recording_time_s=35 + times,
            measured_gt_valid=np.zeros(n, bool),
            trajectory_source="kinematic_estimate_no_physics",
        )
        trajectories[name] = {
            "path": f"{REL}/metadata/object_trajectories/{name}.npz",
            "npz_object_index": len(names) - 1,
            "coordinate_frame": "IKEA_1012_world",
            "quaternion_order": "wxyz",
            "measured": False,
        }
    motion.update(
        object_names=np.asarray(names),
        object_pos_w=np.stack(positions, 1),
        object_quat_w=np.stack(quats, 1),
        object_lin_vel_w=np.stack(linears, 1),
        object_ang_vel_w=np.stack(angulars, 1),
        timestamps=times,
        recording_time_s=35 + times,
        source_motion_frame=np.arange(n),
        object_trajectory_measured=np.asarray(False),
        object_trajectory_source=np.asarray("kinematic_estimate_no_physics"),
    )
    np.savez_compressed(out / "motion.npz", **motion)

    def load_mesh(p, scale=(1, 1, 1)):
        mesh = trimesh.load(p, force="mesh", process=False)
        mesh.vertices *= np.asarray(scale)
        return mesh

    def write_usd(name, pieces, rigid=False):
        path = out / "assets" / (name + ".usd")
        stage = Usd.Stage.CreateNew(str(path))
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        root = UsdGeom.Xform.Define(stage, "/Asset").GetPrim()
        stage.SetDefaultPrim(root)
        if rigid:
            UsdPhysics.RigidBodyAPI.Apply(root)
        count = 0
        for mesh, color, visual, collision in pieces:
            m = UsdGeom.Mesh.Define(stage, f"/Asset/mesh_{count}")
            m.CreatePointsAttr(mesh.vertices.astype(np.float32).tolist())
            m.CreateFaceVertexCountsAttr([3] * len(mesh.faces))
            m.CreateFaceVertexIndicesAttr(mesh.faces.flatten().tolist())
            m.CreateSubdivisionSchemeAttr("none")
            m.CreateDisplayColorAttr([Gf.Vec3f(*map(float, color[:3]))])
            m.CreateDoubleSidedAttr(True)
            if not visual:
                m.CreateVisibilityAttr("invisible")
            if collision:
                UsdPhysics.CollisionAPI.Apply(m.GetPrim())
                UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim()).CreateApproximationAttr(
                    "convexHull" if rigid else "none"
                )
            count += 1
        stage.GetRootLayer().Save()
        reopened = Usd.Stage.Open(str(path))
        assert reopened.GetDefaultPrim()
        return f"{REL}/assets/{name}.usd"

    def body_pieces(body, world=False):
        # These source prop bodies have no nested bodies; avoid silently losing transforms.
        assert not body.findall("body")
        bpos = np.fromstring(body.get("pos", "0 0 0"), sep=" ") if world else np.zeros(3)
        bq = np.fromstring(body.get("quat", "1 0 0 0"), sep=" ") if world else np.array([1, 0, 0, 0])
        br = Rotation.from_quat(bq[[1, 2, 3, 0]])
        for g in body.findall("geom"):
            visual = g.get("group") != "3" and float(g.get("rgba", "1 1 1 1").split()[-1]) > 0
            collision = g.get("contype", "1") != "0"
            if g.get("type") == "mesh":
                p, scale = meshes[g.get("mesh")]
                mesh = load_mesh(p, scale)
            elif g.get("type") == "box":
                mesh = trimesh.creation.box(extents=2 * np.fromstring(g.get("size"), sep=" "))
            else:
                raise ValueError(g.attrib)
            q = np.fromstring(g.get("quat", "1 0 0 0"), sep=" ")
            pos = np.fromstring(g.get("pos", "0 0 0"), sep=" ")
            mesh.vertices = br.apply(Rotation.from_quat(q[[1, 2, 3, 0]]).apply(mesh.vertices) + pos) + bpos
            yield mesh, np.fromstring(g.get("rgba", ".5 .5 .5 1"), sep=" "), visual, collision

    room_pieces = [(load_mesh(config["room_mesh"]), [0.65] * 3, True, True)]
    room_pieces += [
        (load_mesh(i["path"]), [0.6] * 3, True, i.get("collide", True)) for i in config["additional_static_assets"]
    ]
    rack = xmlroot.find("./worldbody/body[@name='coat_rack_support']")
    room_pieces.extend(body_pieces(rack, world=True))
    objects = []
    for index, name in enumerate(names):
        body = xmlroot.find(f"./worldbody/body[@name='{name}']")
        if body is not None:
            usd = write_usd(name, body_pieces(body), rigid=True)
        else:
            spec = next(i for i in config["additional_bodies"] if i["name"] == name)
            pieces = [
                (load_mesh(spec["visual_mesh"]), spec.get("color", [0.5] * 3), True, not spec.get("collision_meshes"))
            ]
            pieces += [(load_mesh(p), [0.5] * 3, False, True) for p in spec.get("collision_meshes", [])]
            usd = write_usd(name, pieces, rigid=True)
        objects.append(
            {
                "id": name,
                "type": name,
                "asset_kind": "rigid_object",
                "physics_mode": "dynamic",
                "usd_path": usd,
                "scale": [1.0, 1.0, 1.0],
                "initial_position_m": positions[index][0].tolist(),
                "initial_orientation_wxyz": quats[index][0].tolist(),
                "initial_position_frame": "world",
                "scene_field": name,
                "prim_name": name,
                "trajectory_kind": "estimated_kinematic_reference",
            }
        )
    # Drawer motion is retained in the exact RSS snapshot and scene_tracks. It is static in this clip.
    drawer = xmlroot.find("./worldbody/body[@name='drawer']")
    dz = np.load(src / "scene_tracks/drawer_slide.npz")
    dump(
        meta / "drawer_fields.json",
        {
            k: {"shape": list(dz[k].shape), "min": float(np.min(dz[k])), "max": float(np.max(dz[k]))}
            for k in dz.files
            if np.issubdtype(dz[k].dtype, np.number)
        },
    )
    assert np.allclose(dz["position"], dz["position"][0])
    drawer_pos = np.fromstring(drawer.get("pos", "0 0 0"), sep=" ")
    drawer_pos += np.fromstring(drawer.find("joint").get("axis"), sep=" ") * float(dz["position"][0])
    drawer.set("pos", " ".join(map(str, drawer_pos)))
    room_pieces.extend(body_pieces(drawer, world=True))
    room_usd = write_usd("room_registered_with_coat_rack_and_drawer", room_pieces)
    config_yaml = {
        "path": f"{REL}/motion.npz",
        "weight": 1.0,
        "motion_fps": fps,
        "source_frame_count": n,
        "duration_seconds": n / fps,
        "source_time_interval_seconds_inclusive": [35, 50],
        "accepted": False,
        "training_data_ready": False,
        "archive_status": "kinematic_final_user_requested_archive",
        "dynamics": "not_run",
        "visual_gt": "not_generated",
        "object_trajectories": trajectories,
        "scene_setup": {
            "runtime_ready": False,
            "status": "kinematic_archive_physics_not_validated",
            "scene": {"usd_path": room_usd, "world_translation_m": [0, 0, 0], "world_orientation_wxyz": [1, 0, 0, 0]},
            "objects": objects,
            "object": next(o for o in objects if o["id"] == "black_ribbon_hat"),
            "exact_rss_scene_config": f"{REL}/scene/config.json",
        },
    }
    schedule = {
        "time_base": "recording_time_s",
        "clip_start_s": 35.0,
        "source": "designed kinematic phases, not measured contacts",
        "close_transition_s": [38.3, 39.05],
        "natural_carry_blend_s": [39.3, 40.0, 44.8, 45.8],
        "release_transition_s": [47.64, 48.1],
        "physics_contact_verified": False,
    }
    dump(meta / "contact_schedule.json", schedule)
    config_yaml["scene_setup"]["hoi_contact_schedule"] = f"{REL}/metadata/contact_schedule.json"
    (out / "motions.yaml").write_text(
        yaml.safe_dump({"hoi1019_hat_to_stand_kinematic_v1": config_yaml}, sort_keys=False)
    )
    source_motion = dict(np.load(src / "motion.npz"))
    assert all(np.array_equal(motion[k], v) for k, v in source_motion.items())
    assert all(np.isfinite(v).all() for v in motion.values() if v.dtype.kind in "fiu")
    assert np.max(np.abs(np.linalg.norm(motion["object_quat_w"], axis=-1) - 1)) < 1e-5
    assert np.max(np.abs(np.linalg.norm(motion["body_quat_w"], axis=-1) - 1)) < 1e-3
    for name, p, q in zip(names, positions, quats, strict=True):
        z = np.load(meta / "object_trajectories" / f"{name}.npz")
        assert np.array_equal(z["position_w"], p) and np.array_equal(z["quaternion_wxyz"], q)
    dump(
        meta / "validation.json",
        {
            "frames": n,
            "fps": fps,
            "robot_source_fields_exact": True,
            "object_sidecars_match": True,
            "finite_and_quaternion_checks": True,
            "usd_files_opened": True,
            "dynamics": "not_run",
            "training_data_ready": False,
        },
    )
    (out / "README.md").write_text(
        "# 1019 帽子抓取、自然搬运与放置\n\n最终归档版本：natural_v4，35–50秒（含末帧），751帧 / 50 Hz。\n\n"
        "motion.npz保留原始43关节/50身体七字段，并加入8个同步物体的世界坐标轨迹（四元数wxyz）。"
        "scene/保存独立网格依赖和原审阅RSS场景；assets/*.usd由当前网格生成，帽子圆柱为旋转90度后的几何。"
        "metadata/保留人体、纯body prior、源帧映射、抽屉轨迹、检查记录和旧ledger。\n\n"
        "这是用户要求保存的最终运动学版本，未进行动力学抓力、滑移、支撑和自由释放验证；无新RGB/DINO。"
        "旧ledger保留当时的搬运待审状态，本次归档请求独立记录，不冒充物理或训练验收。\n"
    )
    dump(
        meta / "provenance.json",
        {
            "source_run": str(run),
            "source_version": "natural_v4",
            "source_motion_sha256": sha(src / "motion.npz"),
            "user_request": "archive final hat result to ScaleTrack/data/1019test",
            "source_ledger_preserved": True,
            "dynamics": "not_run",
            "asset_sources": {str(k): str(v.relative_to(out)) for k, v in copied.items()},
            "files_sha256": {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*")) if p.is_file()},
        },
    )
    print(out)


if __name__ == "__main__":
    main()
