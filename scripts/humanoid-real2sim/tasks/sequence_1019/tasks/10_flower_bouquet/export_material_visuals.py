"""Preserve source USD material colors as OBJ/MTL sections for RSS visuals."""

import argparse
import json
import sys
from pathlib import Path

import bpy

p = argparse.ArgumentParser()
p.add_argument("--run", type=Path, required=True)
a = p.parse_args(sys.argv[sys.argv.index("--") + 1 :])
folder = a.run / "scene/10_flower_bouquet"
dest = folder / "registered_assets"
source = Path("/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/objects/1012")
report = {}
for name in ["flower_bouquet", "small_wicker_basket"]:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.wm.usd_import(filepath=str(source / (name + "_v2.usd")), import_materials=True)
    scale = (
        json.loads((folder / "flower_registered_candidate.json").read_text()).get("native_scale", [1.0, 1.0, 1.0])
        if name == "flower_bouquet"
        else [1.0, 1.0, 1.0]
    )
    if name == "small_wicker_basket":
        dx, dy, dz = json.loads((folder / "basket_registered_candidate.json").read_text())["dimensions_world_aligned_m"]
        scale = [dy / 0.25, dx / 0.20, dz / 0.16]
    colors = {}
    vertices = []
    sections = []
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue
        offset = len(vertices)
        for v in obj.data.vertices:
            xyz = obj.matrix_world @ v.co
            vertices.append([xyz[i] * scale[i] for i in range(3)])
        for f in obj.data.polygons:
            material = obj.material_slots[f.material_index].material if obj.material_slots else None
            key = material.name.replace(" ", "_") if material else "fallback"
            color = list(material.diffuse_color[:3]) if material else [0.5, 0.5, 0.5]
            if material and material.use_nodes:
                shader = next((n for n in material.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
                if shader:
                    color = list(shader.inputs["Base Color"].default_value[:3])
            colors[key] = color
            sections.append((key, [offset + i + 1 for i in f.vertices]))
    with (dest / (name + ".obj")).open("w") as f:
        f.write("mtllib " + name + ".mtl\n")
        for v in vertices:
            f.write("v " + " ".join(map(str, v)) + "\n")
        active = None
        for key, face in sections:
            if key != active:
                f.write("usemtl " + key + "\n")
                active = key
            f.write("f " + " ".join(map(str, face)) + "\n")
    with (dest / (name + ".mtl")).open("w") as f:
        for key, color in colors.items():
            f.write("newmtl " + key + "\nKd " + " ".join(map(str, color)) + "\nKa 0 0 0\nd 1\n")
    report[name] = {
        "materials": colors,
        "vertices": len(vertices),
        "source": str(source / (name + "_v2.usd")),
        "scale": scale,
    }
(dest / "material_visual_provenance.json").write_text(json.dumps(report, indent=2))
print(report)
