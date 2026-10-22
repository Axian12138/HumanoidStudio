"""Apply the reviewed basket parameter candidate to a copied USD with Blender's USD Python."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main(path: Path) -> None:
    from pxr import Gf, Usd, UsdPhysics

    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise ValueError(f"cannot open USD: {path}")
    root = stage.GetPrimAtPath("/Asset")
    mass = UsdPhysics.MassAPI(root)
    mass.GetMassAttr().Set(1.6)
    mass.GetCenterOfMassAttr().Set(Gf.Vec3f(0.0, 0.0, 0.08))
    for prim_path in ("/Asset/PhysicsMaterial", "/Asset/CollisionSet/PhysicsMaterial"):
        material = UsdPhysics.MaterialAPI(stage.GetPrimAtPath(prim_path))
        material.GetStaticFrictionAttr().Set(0.8)
        material.GetDynamicFrictionAttr().Set(0.8)
    stage.GetRootLayer().Save()
    stage = Usd.Stage.Open(str(path))
    root = stage.GetPrimAtPath("/Asset")
    mass = UsdPhysics.MassAPI(root)
    center = list(mass.GetCenterOfMassAttr().Get())
    materials = {}
    for prim_path in ("/Asset/PhysicsMaterial", "/Asset/CollisionSet/PhysicsMaterial"):
        material = UsdPhysics.MaterialAPI(stage.GetPrimAtPath(prim_path))
        materials[prim_path] = {
            "static_friction": float(material.GetStaticFrictionAttr().Get()),
            "dynamic_friction": float(material.GetDynamicFrictionAttr().Get()),
        }
    collision_meshes = sum("PhysicsCollisionAPI" in prim.GetAppliedSchemas() for prim in stage.Traverse())
    report = {
        "mass_kg": float(mass.GetMassAttr().Get()),
        "center_of_mass_local_m": center,
        "materials": materials,
        "collision_meshes": collision_meshes,
    }
    report["verified"] = (
        abs(report["mass_kg"] - 1.6) < 1e-5
        and all(abs(a - b) < 1e-5 for a, b in zip(center, (0, 0, 0.08), strict=True))
        and all(abs(value - 0.8) < 1e-5 for item in materials.values() for value in item.values())
        and collision_meshes == 65
    )
    path.with_suffix(".physics.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["verified"]:
        raise ValueError(report)


if __name__ == "__main__":
    main(Path(sys.argv[sys.argv.index("--") + 1]))
