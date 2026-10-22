"""Export source USD geometry and its measured drawer to a task-local MJCF.

The drawer's visual meshes are removed from the static room and retained on
its +Y slider. Source USD transforms are evaluated, never inferred from names.
"""

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ST = Path("/home/galbot/workspace/ScaleTrack")

ASSETS = ST / "source/scaletrack/scaletrack/assets"
ROOM = ASSETS / "scenes/tianze_sim_data/IKEA_LivingRoom_Mesh_1012"


def export_meshes(
    path, dest, excluded_prefix=None, included_prefix=None, offset=None, include_hidden=False, translations=None
):
    from pxr import Gf, Usd, UsdGeom

    stage = Usd.Stage.Open(str(path))
    cache = UsdGeom.XformCache()
    vertices, faces, prims = [], [], []
    for prim in stage.Traverse():
        name = str(prim.GetPath())
        if not prim.IsA(UsdGeom.Mesh) or (excluded_prefix and name.startswith(excluded_prefix)):
            continue
        if included_prefix and not name.startswith(included_prefix):
            continue
        imageable = UsdGeom.Imageable(prim)
        if not include_hidden and (
            imageable.ComputeVisibility() == "invisible" or imageable.ComputePurpose() in ("guide", "proxy")
        ):
            continue
        mesh = UsdGeom.Mesh(prim)
        mat = cache.GetLocalToWorldTransform(prim)
        vv = np.asarray([mat.Transform(Gf.Vec3d(*map(float, x))) for x in mesh.GetPointsAttr().Get()])
        for prefix, translation in (translations or {}).items():
            if name.startswith(prefix):
                vv += np.asarray(translation)
        if offset is not None:
            vv -= offset
        index = len(vertices)
        vertices.extend(vv.tolist())
        ids = list(mesh.GetFaceVertexIndicesAttr().Get())
        cursor = 0
        for count in mesh.GetFaceVertexCountsAttr().Get():
            polygon = ids[cursor : cursor + count]
            for j in range(1, count - 1):
                faces.append([index + polygon[k] + 1 for k in (0, j, j + 1)])
            cursor += count
        prims.append(name)
    with dest.open("w") as f:
        for v in vertices:
            f.write("v " + " ".join(map(str, v)) + "\n")
        for face in faces:
            f.write("f " + " ".join(map(str, face)) + "\n")
    return {
        "source": str(path),
        "path": str(dest),
        "prim_paths": prims,
        "bounds_m": [np.min(vertices, axis=0).tolist(), np.max(vertices, axis=0).tolist()],
    }


def main():
    import importlib

    import mujoco

    sys.path.insert(0, str(ST / "retarget3/src"))
    importlib.import_module("retarget3.runtime")
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    out = args.run / "scene/03_04_05"
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOM / "drawer_1012_v4_manifest.json").read_text())
    source = ROOM / "IKEA_LivingRoom_Mesh_1012_drawer_renderable_v4.usd"
    moving = "/World/MeasuredCabinetDrawer/Drawer"
    records = {}
    records["room"] = export_meshes(source, out / "room_without_drawer.obj", excluded_prefix=moving)
    records["drawer"] = export_meshes(
        source, out / "drawer.obj", included_prefix=moving + "/Visual", offset=np.array(manifest["open_origin_m"])
    )
    for name, file in [
        ("floor_strawberry", "../raw_assets/PlushToy_Strawberry/PlushToy_Strawberry_policyplay_20260917.usda"),
        ("black_ribbon_hat", "black_ribbon_hat_v2.usd"),
        ("round_laundry_basket", "round_laundry_basket_v2.usd"),
    ]:
        records[name] = export_meshes(ASSETS / "objects/1012" / file, out / (name + ".obj"))
    prop_root = ET.Element("mujoco", model="sequence_1019_props")
    prop_asset = ET.SubElement(prop_root, "asset")
    prop_world = ET.SubElement(prop_root, "worldbody")
    prop_sources = {
        "floor_strawberry": ASSETS
        / "objects/raw_assets/PlushToy_Strawberry/PlushToy_Strawberry_policyplay_20260917.usda",
        "black_ribbon_hat": ASSETS / "objects/1012/black_ribbon_hat_v2.usd",
        "round_laundry_basket": ASSETS / "objects/1012/round_laundry_basket_v2.usd",
    }
    initial = {
        "floor_strawberry": [0.65, -0.05, 0.0367203],
        "black_ribbon_hat": [2.005, 0.82, 0.4431],
        "round_laundry_basket": [0.07475285, -2.13050008, 0.001],
    }
    for name, source_path in prop_sources.items():
        body = ET.SubElement(prop_world, "body", name=name, pos=" ".join(map(str, initial[name])))
        ET.SubElement(body, "freejoint", name=name + "_free")
        ET.SubElement(prop_asset, "mesh", name=name + "_visual", file=str((out / (name + ".obj")).resolve()))
        stage = Usd.Stage.Open(str(source_path))
        records[name]["visual_materials"] = []
        for visual_index, prim in enumerate(stage.Traverse()):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            imageable = UsdGeom.Imageable(prim)
            if imageable.ComputeVisibility() == "invisible" or imageable.ComputePurpose() in ("guide", "proxy"):
                continue
            visual_name = name + "_visual_part_" + str(visual_index)
            visual_path = out / (visual_name + ".obj")
            export_meshes(source_path, visual_path, included_prefix=str(prim.GetPath()))
            color = [0.5, 0.5, 0.5]
            material = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
            if material:
                for shader_prim in material.GetPrim().GetChildren():
                    shader = UsdShade.Shader(shader_prim)
                    if shader and shader.GetInput("diffuseColor"):
                        diffuse = shader.GetInput("diffuseColor").Get()
                        if diffuse is not None:
                            color = list(map(float, diffuse))
            records[name]["visual_materials"].append(
                {
                    "prim": str(prim.GetPath()),
                    "material": str(material.GetPath()) if material else None,
                    "diffuse_rgb": color,
                    "texture_transferred": False,
                }
            )
            ET.SubElement(prop_asset, "mesh", name=visual_name, file=str(visual_path.resolve()))
            ET.SubElement(
                body,
                "geom",
                type="mesh",
                mesh=visual_name,
                contype="0",
                conaffinity="0",
                group="2",
                mass="0",
                rgba=" ".join(map(str, [*color, 1])),
            )
        collision = [p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh) and p.HasAPI(UsdPhysics.CollisionAPI)]
        records[name]["collision_prim_paths"] = [str(p.GetPath()) for p in collision]
        for i, prim in enumerate(collision):
            mesh_name = name + "_collision_" + str(i)
            mesh_path = out / (mesh_name + ".obj")
            export_meshes(source_path, mesh_path, included_prefix=str(prim.GetPath()), include_hidden=True)
            ET.SubElement(prop_asset, "mesh", name=mesh_name, file=str(mesh_path.resolve()))
            ET.SubElement(
                body,
                "geom",
                type="mesh",
                mesh=mesh_name,
                group="3",
                rgba="0 0 0 0",
                mass=str((1.6 if name == "round_laundry_basket" else 0.1) / max(len(collision), 1)),
            )
        if not collision:
            records[name]["collision_limitation"] = (
                "source has no authored collision mesh; visual convex hull used as explicit provisional proxy"
            )
            ET.SubElement(body, "geom", type="mesh", mesh=name + "_visual", group="3", rgba="0 0 0 0", mass=".1")
    ET.ElementTree(prop_root).write(out / "props.xml", encoding="unicode")
    root = ET.Element("mujoco", model="sequence_1019_drawer")
    ET.SubElement(root, "compiler", angle="radian")
    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "mesh", name="drawer_visual", file=str(out.resolve() / "drawer.obj"))
    world = ET.SubElement(root, "worldbody")
    body = ET.SubElement(world, "body", name="drawer", pos=" ".join(map(str, manifest["closed_origin_m"])))
    ET.SubElement(
        body, "joint", name="drawer_slide", type="slide", axis="0 1 0", limited="true", range="0 0.19", damping="0.5"
    )
    ET.SubElement(
        body,
        "geom",
        type="mesh",
        mesh="drawer_visual",
        contype="0",
        conaffinity="0",
        mass="0",
        group="2",
        rgba="0.7 0.65 0.55 1",
    )
    for part in manifest["moving_parts"]:
        lo, hi = np.asarray(part["initial_world_bounds_m"])
        center = (lo + hi) / 2 - manifest["open_origin_m"]
        ET.SubElement(
            body,
            "geom",
            name="drawer_" + part["name"],
            type="box",
            pos=" ".join(map(str, center)),
            size=" ".join(map(str, (hi - lo) / 2)),
            group="3",
            rgba="0 0 0 0",
            mass=str(4 / len(manifest["moving_parts"])),
        )
    ET.ElementTree(root).write(out / "drawer.xml", encoding="unicode")

    model = mujoco.MjModel.from_xml_path(str(out / "drawer.xml"))
    data = mujoco.MjData(model)
    data.qpos[0] = 0.19
    mujoco.mj_forward(model, data)
    initial = float(data.qpos[0])
    for _ in range(1000):
        mujoco.mj_step(model, data)
    records["drawer_validation"] = {
        "joint_name": "drawer_slide",
        "axis": [0, 1, 0],
        "limits_m": [0, 0.19],
        "initial_q_m": initial,
        "after_2s_q_m": float(data.qpos[0]),
        "no_robot": True,
        "backend": "MuJoCo independent mechanism test; RSS live review pending",
    }
    robot_source = ASSETS / "robots/g1_29dof_dex3/g1_29dof_dex3_offsethand.xml"
    merged = ET.parse(robot_source).getroot()
    merged.find("compiler").set("meshdir", str(robot_source.parent / "meshes"))
    for extra in (root, prop_root):
        for item in extra.find("asset"):
            merged.find("asset").append(item)
        for item in extra.find("worldbody"):
            merged.find("worldbody").append(item)
    ET.ElementTree(merged).write(out / "robot_props_drawer.xml", encoding="unicode")
    mujoco.MjModel.from_xml_path(str(out / "robot_props_drawer.xml"))
    records["drawer_source_manifest"] = str(ROOM / "drawer_1012_v4_manifest.json")
    records["static_drawer_removed"] = not any(p.startswith(moving) for p in records["room"]["prim_paths"])
    (out / "manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()
