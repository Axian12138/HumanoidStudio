"""Export the authored USD meshes in native asset coordinates, without axis changes."""

import argparse
import json
import sys
from pathlib import Path

import bpy

p = argparse.ArgumentParser()
p.add_argument("--out", required=True)
a = p.parse_args(sys.argv[sys.argv.index("--") + 1 :])
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
source = Path("/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/objects/1012")
report = {}
for name in ["flower_bouquet", "small_wicker_basket"]:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.wm.usd_import(filepath=str(source / f"{name}_v2.usd"), import_materials=False)
    vertices = []
    faces = []
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue
        mesh = obj.data
        offset = len(vertices)
        vertices.extend([tuple(obj.matrix_world @ v.co) for v in mesh.vertices])
        faces.extend([tuple(offset + i + 1 for i in f.vertices) for f in mesh.polygons])
    dest = out / f"{name}.obj"
    with dest.open("w") as f:
        for v in vertices:
            f.write("v " + " ".join(map(str, v)) + "\n")
        for face in faces:
            f.write("f " + " ".join(map(str, face)) + "\n")
    report[name] = {
        "source": str(source / f"{name}_v2.usd"),
        "vertices": len(vertices),
        "bounds": [[min(v[i] for v in vertices) for i in range(3)], [max(v[i] for v in vertices) for i in range(3)]],
    }
(out / "asset_report.json").write_text(json.dumps(report, indent=2))
