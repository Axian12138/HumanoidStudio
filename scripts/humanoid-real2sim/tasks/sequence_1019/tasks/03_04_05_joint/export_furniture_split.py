"""Export early furniture in Motive world, leaving a remainder for late registration."""

import argparse
import importlib
import json
import sys
from pathlib import Path

from build_scene import ROOM, ST, export_meshes


def main():
    import mujoco  # load the repository runtime before the USD environment

    _ = mujoco
    sys.path.insert(0, str(ST / "retarget3/src"))
    importlib.import_module("retarget3.runtime")
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    a = p.parse_args()
    base = a.run / "scene/03_04_05"
    out = base / "furniture_split"
    out.mkdir(exist_ok=True)
    prefixes = {
        "rack": "/World/MeasuredFurniture/CoatRack/",
        "sofa": "/World/Origin/SceneOffset/IKEA_LivingRoom_Scene/LivingRoom/LANDSKRONA_3P",
        "drawer": "/World/MeasuredCabinetDrawer/Drawer",
    }
    shift = json.loads((base / "early_rack_registration.json").read_text())["translation_world_m"]
    source = ROOM / "IKEA_LivingRoom_Mesh_1012_drawer_renderable_v4.usd"
    records = {}
    for name in ["rack", "sofa"]:
        translations = {prefixes[name]: shift} if name == "rack" else None
        records[name] = export_meshes(
            source, out / f"{name}_world.obj", included_prefix=prefixes[name], translations=translations
        )
        records[name]["transform_already_baked"] = True
        records[name]["translation_world_m"] = shift if name == "rack" else [0, 0, 0]
    records["remainder"] = export_meshes(
        source, out / "room_without_early_furniture.obj", excluded_prefix=tuple(prefixes.values())
    )
    records["remainder"]["registration"] = (
        "untransformed source; parent may split or register late furniture, avoid applying early furniture shifts"
    )
    report = {
        "status": "diagnostic_registration_not_final",
        "frame": "Motive canonical world; floor z0",
        "parts": records,
        "drawer": "slider remains in robot_props_drawer_with_rack_early_registered.xml at original world position; independent early alignment residual unresolved",
        "robot_scene": str((base / "robot_props_drawer_with_rack_early_registered.xml").resolve()),
        "integration": "Load each furniture once. rack/sofa OBJ positions are baked world coordinates; do not apply late floor_v3 transform to these objects. Remainder excludes rack, sofa, moving drawer only; drawer cabinet carcass remains for separate assessment.",
    }
    (out / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(out / "manifest.json")


if __name__ == "__main__":
    main()
