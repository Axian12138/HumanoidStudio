"""Apply the early image-derived rack translation to separate diagnostic assets."""

import argparse
import importlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from build_scene import ROOM, ST, export_meshes


def main():
    import mujoco

    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    a = p.parse_args()
    base = a.run / "scene/03_04_05"
    fit = json.loads((base / "early_rack_registration.json").read_text())
    shift = np.array(fit["translation_world_m"])
    for name in ("robot_props_drawer_with_rack", "hat_rack_physics"):
        tree = ET.parse(base / f"{name}.xml")
        moved = 0
        for geom in tree.getroot().iter("geom"):
            if geom.get("name", "").startswith("rack_"):
                pos = np.fromstring(geom.get("pos", "0 0 0"), sep=" ")
                geom.set("pos", " ".join(map(str, pos + shift)))
                moved += 1
        if moved != 306:
            raise ValueError(f"{name}: expected 306 rack pieces, found {moved}")
        destination = base / f"{name}_early_registered.xml"
        tree.write(destination, encoding="unicode")
        mujoco.MjModel.from_xml_path(str(destination))
    sys.path.insert(0, str(ST / "retarget3/src"))
    importlib.import_module("retarget3.runtime")
    source = ROOM / "IKEA_LivingRoom_Mesh_1012_drawer_renderable_v4.usd"
    record = export_meshes(
        source,
        base / "room_without_drawer_early_registered.obj",
        excluded_prefix="/World/MeasuredCabinetDrawer/Drawer",
        translations={"/World/MeasuredFurniture/CoatRack/": shift},
    )
    record.update(status="diagnostic_local_rack_registration_only", translation_world_m=shift.tolist())
    (base / "registered_rack_assets.json").write_text(json.dumps(record, indent=2) + "\n")
    print(record["status"], shift.tolist())


if __name__ == "__main__":
    main()
