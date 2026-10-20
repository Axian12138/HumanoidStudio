"""Export the source Blender visual meshes and authored convex collision hulls."""

import json
import sys
from pathlib import Path

import bpy


out = Path(sys.argv[sys.argv.index("--") + 1])
out.mkdir(parents=True, exist_ok=True)
meshes = [obj for obj in bpy.data.objects if obj.type == "MESH"]
visuals = [obj for obj in meshes if not obj.hide_render]
hulls = [obj for obj in meshes if obj.name.startswith("Hull")]
if not visuals or not hulls:
    raise RuntimeError("source asset needs visual meshes and authored collision hulls")


def export(objects, path):
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.hide_set(False)
        obj.hide_viewport = False
        obj.select_set(True)
    bpy.ops.wm.obj_export(
        filepath=str(path), export_selected_objects=True, apply_modifiers=True,
        export_materials=False, forward_axis="Y", up_axis="Z",
    )


export(visuals, out / "visual.obj")
collision_dir = out / "collision_hulls"
collision_dir.mkdir(exist_ok=True)
for hull in hulls:
    export([hull], collision_dir / f"{hull.name}.obj")
(out / "manifest.json").write_text(json.dumps({
    "source_blend": bpy.data.filepath,
    "visual_meshes": [obj.name for obj in visuals],
    "collision_hulls": [obj.name for obj in hulls],
}, indent=2) + "\n")
