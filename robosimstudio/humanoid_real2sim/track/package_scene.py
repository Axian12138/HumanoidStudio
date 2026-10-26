"""Build repair scenes from the current ScaleTrack YAML and composed USD assets.

Only boolean rigid-body/collision semantics are audited for parity. Contact
materials and mesh approximation remain subject to Newton/MuJoCo conversion.
The source package and all referenced assets are read-only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml


def read_package(package: Path) -> dict:
    manifest = package / "motions.yaml"
    entries = yaml.safe_load(manifest.read_text())
    if not isinstance(entries, dict) or len(entries) != 1:
        raise ValueError(f"{manifest}: expected exactly one motion")
    entry = next(iter(entries.values()))
    setup = entry["scene_setup"]
    objects = setup.get("objects") or [setup["object"]]
    ids = [obj["id"] for obj in objects]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate object ids")
    for obj in objects:
        if "physics_mode" in obj and obj["physics_mode"] not in ("dynamic", "kinematic"):
            raise ValueError(f"{obj['id']}: unsupported physics_mode")
        if "collision_enabled" in obj and type(obj["collision_enabled"]) is not bool:
            raise ValueError(f"{obj['id']}: collision_enabled must be a YAML boolean")
    return {"entry": entry, "setup": setup, "objects": objects, "manifest": str(manifest)}


def compose_stage(package: Path, scaletrack: Path, out: Path) -> tuple[Path, dict]:
    """Reference original assets, then author placement and explicit YAML overrides."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    data = read_package(package)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "scene.usda"
    stage = Usd.Stage.CreateNew(str(path))
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    contract = {"manifest": data["manifest"], "assets": [], "bodies": {}, "colliders": {}}
    room = data["setup"]["scene"]
    assets = [("room", room, True), *((o["id"], o, False) for o in data["objects"])]
    for name, cfg, is_room in assets:
        if not Sdf.Path.IsValidIdentifier(name):
            raise ValueError(f"invalid USD object id: {name}")
        source = Path(cfg["usd_path"])
        if not source.is_absolute():
            source = scaletrack / source
        source = source.resolve(strict=True)
        original = Usd.Stage.Open(str(source))
        if not original or not original.GetDefaultPrim():
            raise ValueError(f"{source}: no default prim")
        # Reference-only variant layers often omit stage metadata. Like Isaac's
        # UsdFileCfg references, they inherit the destination's metre/Z-up frame,
        # not Usd.Stage.Open's unauthored centimetre/Y-up fallback.
        if (original.HasAuthoredMetadata("upAxis") and UsdGeom.GetStageUpAxis(original) != UsdGeom.Tokens.z) or (
            original.HasAuthoredMetadata("metersPerUnit")
            and not np.isclose(UsdGeom.GetStageMetersPerUnit(original), 1.0)
        ):
            raise ValueError(f"{source}: only metre/Z-up assets are supported")
        prim = stage.DefinePrim(f"/World/{name}", "Xform")
        prim.GetReferences().AddReference(str(source))
        # Resolve instance proxies before applying per-body/shape overrides.
        for child in list(Usd.PrimRange(prim, Usd.TraverseInstanceProxies())):
            if child.IsInstance():
                child.SetInstanceable(False)
        pos = cfg.get("world_translation_m", (0, 0, 0)) if is_room else cfg["initial_position_m"]
        quat = cfg.get("world_orientation_wxyz", (1, 0, 0, 0)) if is_room else cfg["initial_orientation_wxyz"]
        scale = (1, 1, 1) if is_room else cfg.get("scale", (1, 1, 1))
        if not np.all(np.isfinite([*pos, *quat, *scale])) or min(scale) <= 0:
            raise ValueError(f"{name}: invalid transform")
        quat = np.asarray(quat, dtype=float)
        if not np.isclose(np.linalg.norm(quat), 1, atol=1e-4):
            raise ValueError(f"{name}: orientation is not a unit quaternion")
        xf = UsdGeom.Xformable(prim)
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(*pos))
        xf.AddOrientOp().Set(Gf.Quatf(float(quat[0]), Gf.Vec3f(*quat[1:])))
        xf.AddScaleOp().Set(Gf.Vec3f(*scale))
        body_count = 0
        collider_count = 0
        for child in Usd.PrimRange(prim):
            key = str(child.GetPath())
            if child.HasAPI(UsdPhysics.RigidBodyAPI):
                api = UsdPhysics.RigidBodyAPI(child)
                if not is_room and "physics_mode" in cfg:
                    api.CreateKinematicEnabledAttr().Set(cfg["physics_mode"] == "kinematic")
                if not api.GetRigidBodyEnabledAttr().Get():
                    raise ValueError(f"{key}: disabled rigid body is unsupported; use static geometry")
                contract["bodies"][key] = bool(api.GetKinematicEnabledAttr().Get())
                body_count += 1
            if child.HasAPI(UsdPhysics.CollisionAPI):
                api = UsdPhysics.CollisionAPI(child)
                if not is_room and "collision_enabled" in cfg:
                    api.CreateCollisionEnabledAttr().Set(cfg["collision_enabled"])
                contract["colliders"][key] = bool(api.GetCollisionEnabledAttr().Get())
                collider_count += 1
        if not is_room and body_count != 1:
            raise ValueError(f"{name}: expected one rigid body, got {body_count}")
        if not is_room and cfg.get("collision_enabled") is True and not collider_count:
            raise ValueError(f"{name}: collision enabled but USD has no colliders")
        contract["assets"].append(
            {
                "id": name,
                "usd": str(source),
                "physics_mode_override": cfg.get("physics_mode"),
                "collision_override": cfg.get("collision_enabled"),
                "body_count": body_count,
                "collider_count": collider_count,
                "position": list(pos),
                "scale": list(scale),
            }
        )
    stage.GetRootLayer().Save()
    contract["joint_initial_positions"] = {}
    for articulation in room.get("articulations", []):
        root = articulation.get("relative_prim_path")
        if not root:
            raise ValueError("room articulation needs relative_prim_path")
        for joint, value in articulation.get("initial_joint_position_m", {}).items():
            contract["joint_initial_positions"][f"/World/room/{root}/{joint}"] = float(value)
    (out / "source_contract.json").write_text(json.dumps(contract, indent=2))
    return path, contract


def audit_model(model, contract: dict) -> dict:
    """Reject missing bodies/colliders or changes to effective collision flags."""
    import newton

    def values(x):
        return x.numpy() if hasattr(x, "numpy") else np.asarray(x)

    labels = list(model.body_label)
    body_flags = values(model.body_flags)
    shape_flags = values(model.shape_flags)
    shapes = list(model.shape_label)
    errors = []
    bodies = []
    colliders = []
    for path, expected in contract["bodies"].items():
        indices = [i for i, label in enumerate(labels) if label == path]
        if len(indices) != 1:
            errors.append(f"{path}: expected one body, found {len(indices)}")
            continue
        actual = bool(int(body_flags[indices[0]]) & int(newton.BodyFlags.KINEMATIC))
        bodies.append({"path": path, "index": indices[0], "kinematic": actual})
        if actual != expected:
            errors.append(f"{path}: kinematic={actual}, expected {expected}")
    for path, expected in contract["colliders"].items():
        indices = [i for i, label in enumerate(shapes) if label == path or label.startswith(path + "/")]
        if not indices and expected:
            errors.append(f"{path}: enabled collider missing from imported model")
        actual = [bool(int(shape_flags[i]) & int(newton.ShapeFlags.COLLIDE_SHAPES)) for i in indices]
        # Importers may append an explicitly visual copy of a colliding mesh.
        if (expected and not any(actual)) or (not expected and any(actual)):
            errors.append(f"{path}: collision flags {actual}, expected {expected}")
        colliders.append({"path": path, "indices": indices, "enabled": actual, "expected": expected})
    # No geometry without an enabled USD collider may acquire physical contact.
    for i, path in enumerate(shapes):
        if not path.startswith("/World/") or not int(shape_flags[i]) & int(newton.ShapeFlags.COLLIDE_SHAPES):
            continue
        ancestors = [p for p in contract["colliders"] if path == p or path.startswith(p + "/")]
        if not ancestors or not contract["colliders"][max(ancestors, key=len)]:
            errors.append(f"{path}: unexpected collision shape")
    result = {"status": "fail" if errors else "pass", "bodies": bodies, "colliders": colliders, "errors": errors}
    return result


def build_package_sim(asset: Path, motion, *, package: Path, scaletrack: Path, out: Path):
    import robosimstudio as rss

    path, contract = compose_stage(package, scaletrack, out)
    robot = rss.EmbodimentSpec(
        name="g1_package_repair",
        asset=rss.AssetSpec(path=str(asset)),
        preserve_free_bodies=True,
        enable_self_collisions=False,
        arms=(
            rss.ArmSpec(
                name="whole_body",
                joint_names=motion.joint_names,
                ee_body_name="right_wrist_yaw_link",
                home_q=tuple(motion.joint_pos[0]),
            ),
        ),
    )
    # The authored room floor contains planar mesh colliders, which MuJoCo's
    # native collision generator refuses. Keep those colliders and let Newton
    # generate contacts for the MuJoCo solver instead of inventing thick boxes.
    physics = rss.MUJOCO_500HZ.with_backend(
        "*", rss.backend("mujoco", use_mujoco_contacts=False, nconmax=2048, njmax=4096)
    )
    scene = rss.Scene(cell="void", name="package_repair", physics=physics)
    scene.add(rss.robot(robot))
    scene.add(
        rss.StaticSpec(
            name="package_scene", asset=rss.AssetSpec(path=str(path)), world_xform=True, expose_declared_colliders=True
        )
    )
    sim = scene.build(control_mode="joint_position")
    report = audit_model(sim.model, contract)
    report["source_contract"] = str(out / "source_contract.json")
    report["limitations"] = [
        "Contact materials, convex approximation and solver response are not Isaac/PhysX equivalent.",
        "Kinematic objects retain their initial pose; reference object trajectories are not replayed.",
    ]
    (out / "physics_audit.json").write_text(json.dumps(report, indent=2))
    if report["errors"]:
        raise ValueError("USD physics audit failed: " + "; ".join(report["errors"][:8]))
    q = sim.pipeline.snapshot.joint_q.copy()
    starts = sim.model.joint_q_start.numpy()
    for name, value in contract["joint_initial_positions"].items():
        if name not in sim.model.joint_label:
            raise ValueError(f"missing room joint {name}")
        q[int(starts[list(sim.model.joint_label).index(name)])] = value
    sim._start_joint_q = q
    sim.reset()
    return sim


def check_steps(sim, out: Path, steps: int = 5) -> dict:
    """Short physical smoke check; never a claim that the motion succeeds."""
    import newton

    flags = sim.model.body_flags.numpy()
    locked = np.flatnonzero(flags & int(newton.BodyFlags.KINEMATIC))
    initial = sim.pipeline.state_0.body_q.numpy().copy()
    max_drift = 0.0
    for _ in range(steps):
        sim.run(1).sync()
        current = sim.pipeline.state_0.body_q.numpy()
        if not np.all(np.isfinite(current)):
            raise ValueError("non-finite state during scene smoke check")
        if len(locked):
            max_drift = max(max_drift, float(np.max(np.abs(current[locked] - initial[locked]))))
        contacts = sim.pipeline.contacts
        count = int(contacts.rigid_contact_count.numpy()[0])
        shape_ids = np.concatenate(
            [contacts.rigid_contact_shape0.numpy()[:count], contacts.rigid_contact_shape1.numpy()[:count]]
        )
        shape_ids = shape_ids[shape_ids >= 0]
        collision_flags = sim.model.shape_flags.numpy()[shape_ids]
        if np.any((collision_flags & int(newton.ShapeFlags.COLLIDE_SHAPES)) == 0):
            raise ValueError("disabled shape appeared in physical contacts")
    if max_drift > 1e-6:
        raise ValueError(f"kinematic body moved during physical stepping: {max_drift}")
    report = {
        "status": "pass",
        "steps": steps,
        "kinematic_bodies": len(locked),
        "maximum_pose_component_drift": max_drift,
        "disabled_shape_contacts": 0,
        "scope": "short scene smoke check, not tracker or full-motion validation",
    }
    (out / "step_check.json").write_text(json.dumps(report, indent=2))
    sim.reset()
    return report
