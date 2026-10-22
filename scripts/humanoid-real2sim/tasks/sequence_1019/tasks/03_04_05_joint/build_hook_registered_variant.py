"""Preserve the rack base/axis while correcting independently observed lowest hooks."""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def main():
    from scipy.spatial.transform import Rotation

    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--empty45", action="store_true")
    p.add_argument("--multiframe", action="store_true")
    p.add_argument("--carry-camera", action="store_true")
    a = p.parse_args()
    r = a.run
    base = r / "scene/03_04_05"
    fit = json.loads((r / "review/scene/03_04_05_rack_hook_geometry_fit.json").read_text())["fit48"]
    axis = np.array(json.loads((r / "review/scene/03_04_05_rack_hook_yaw_fit.json").read_text())["axis_world_m"])
    suffix = "hook_registered"
    xy_delta = np.zeros(3)
    if a.empty45:
        fit45 = json.loads((r / "review/scene/03_04_05_rack_empty45_fit.json").read_text())
        fit, axis = fit45["fit"], np.array(fit45["fixed_axis_world_m"])
        suffix = "empty45_hook_diagnostic"
    if a.multiframe:
        combined = json.loads((r / "review/scene/03_04_05_rack_multiframe_local_camera_fit.json").read_text())
        fit = combined["fit"]
        xy_delta[:2] = combined["xy_delta_m"]
        suffix = "multiframe_hook_diagnostic"
    if a.carry_camera:
        combined = json.loads((r / "review/scene/03_04_05_rack_carry_camera_fit.json").read_text())
        fit = combined["fit"]
        xy_delta[:2] = combined["xy_delta_m"]
        suffix = "carry_camera_hook_diagnostic"
    R = Rotation.from_euler("z", fit["yaw_rad"])
    scale = np.array([fit["radial_scale"], fit["radial_scale"], 1])
    shift = np.array([0, 0, fit["height_offset_m"]])
    parts = json.loads((base / "rack_support.json").read_text())["parts"]
    selected = [
        p
        for p in parts
        if any(f"hook_0_{i}_" in p["source_prim"] for i in range(3))
        or p["source_prim"].endswith(
            ("part_0030_CoatRack_hook_ball", "part_0055_CoatRack_hook_ball_001", "part_0080_CoatRack_hook_ball_002")
        )
    ]
    names = {p["geom"] for p in selected}
    outputs = []
    for src, dst in [
        ("hat_rack_physics_early_registered.xml", f"hat_rack_physics_{suffix}.xml"),
        ("robot_props_drawer_with_rack_early_registered.xml", f"robot_props_drawer_with_rack_{suffix}.xml"),
    ]:
        tree = ET.parse(base / src)
        root = tree.getroot()
        body = root.find('.//body[@name="coat_rack_support"]')
        for geom in body.findall("geom"):
            original_pos = np.fromstring(geom.get("pos", "0 0 0"), sep=" ")
            if geom.get("name") not in names:
                if a.multiframe or a.carry_camera:
                    geom.set("pos", " ".join(map(str, original_pos + xy_delta)))
                continue
            pos = original_pos
            geom.set("pos", " ".join(map(str, R.apply((pos - axis) * scale) + axis + shift + xy_delta)))
            old = np.fromstring(geom.get("quat", "1 0 0 0"), sep=" ")
            quat = (R * Rotation.from_quat(old[[1, 2, 3, 0]])).as_quat()[[3, 0, 1, 2]]
            geom.set("quat", " ".join(map(str, quat)))
            mesh = root.find(f'.//asset/mesh[@name="{geom.get("mesh")}"]')
            mesh.set("scale", " ".join(map(str, scale)))
        tree.write(base / dst)
        outputs.append(dst)
    A = R.as_matrix() @ np.diag(scale)
    T = np.eye(4)
    T[:3, :3] = A
    T[:3, 3] = axis + shift + xy_delta - A @ axis
    report = {
        "status": "lowest_hook_variant_pending_independent_holdout_and_physics",
        "source_fit": "review/scene/03_04_05_rack_carry_camera_fit.json"
        if a.carry_camera
        else "review/scene/03_04_05_rack_multiframe_local_camera_fit.json"
        if a.multiframe
        else (
            "review/scene/03_04_05_rack_empty45_fit.json"
            if a.empty45
            else "review/scene/03_04_05_rack_hook_geometry_fit.json"
        ),
        "selected_parts": selected,
        "selected_count": len(selected),
        "T_after_early_translation": T.tolist(),
        "outputs": outputs,
        "scope": "Lowest three curved hooks/end balls transformed jointly; base/mast/upperhooks receive common XY translation only in multiframe/carry variants. Hidden ball is inferred by authored common geometry, not independently observed.",
        "carry_camera": a.carry_camera,
        "empty45": a.empty45,
        "multiframe": a.multiframe,
        "whole_rack_xy_delta_m": xy_delta[:2].tolist(),
    }
    (base / f"{suffix}_variant.json").write_text(json.dumps(report, indent=2) + "\n")
    print({"selected_count": len(selected), "outputs": outputs})


if __name__ == "__main__":
    main()
