"""Run with Blender to export the basket visual and authored 65 collision hulls.

blender -b round_laundry_basket_v2.blend -P export_assets.py -- OUTPUT_DIR
"""

import sys
from pathlib import Path

import bpy


def export(objects, path):
    for obj in bpy.data.objects:
        obj.select_set(False)
    for obj in objects:
        obj.hide_set(False)
        obj.hide_viewport = False
        obj.select_set(True)
    bpy.ops.wm.obj_export(
        filepath=str(path),
        export_selected_objects=True,
        apply_modifiers=True,
        export_materials=False,
        forward_axis="Y",
        up_axis="Z",
    )


out = Path(sys.argv[sys.argv.index("--") + 1])
out.mkdir(parents=True, exist_ok=True)
meshes = [obj for obj in bpy.data.objects if obj.type == "MESH"]
visuals = [obj for obj in meshes if not obj.hide_render]
hulls = [obj for obj in meshes if obj.name.startswith("Hull")]
if len(hulls) != 65:
    raise ValueError(f"expected 65 source collision hulls, found {len(hulls)}")
export(visuals, out / "round_laundry_basket_v2.obj")
export(hulls, out / "round_laundry_basket_v2_hulls.obj")
collision_dir = out / "collision_hulls"
collision_dir.mkdir(exist_ok=True)
for hull in hulls:
    export([hull], collision_dir / f"{hull.name}.obj")
print(f"exported {len(visuals)} visual meshes and {len(hulls)} collision hulls")
