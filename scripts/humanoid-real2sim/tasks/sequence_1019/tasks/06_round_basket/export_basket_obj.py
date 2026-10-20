"""Export the reviewed 1012 round basket visual meshes in local Z-up metres."""

from pathlib import Path
from robosimstudio.humanoid_real2sim.paths import example_dir

import bpy


HERE = example_dir(__file__)
SOURCE = Path(
    "/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/"
    "objects/1012/round_laundry_basket_v2.blend"
)
OUTPUT = HERE / "round_laundry_basket_visual.obj"
COLLISION = HERE / "round_laundry_basket_collision.obj"

bpy.ops.wm.open_mainfile(filepath=str(SOURCE))
bpy.ops.object.select_all(action="DESELECT")
meshes = [obj for obj in bpy.data.objects if obj.type == "MESH" and not obj.name.startswith("Hull_")]
for obj in meshes:
    obj.select_set(True)
bpy.context.view_layer.objects.active = meshes[0]
bpy.ops.wm.obj_export(
    filepath=str(OUTPUT), export_selected_objects=True, apply_modifiers=True,
    forward_axis="Y", up_axis="Z", export_materials=False,
)
bpy.ops.object.select_all(action="DESELECT")
hulls = [obj for obj in bpy.data.objects if obj.type == "MESH" and obj.name.startswith("Hull_")]
for obj in hulls:
    obj.hide_set(False)
    obj.hide_viewport = False
    obj.hide_render = False
    obj.select_set(True)
bpy.context.view_layer.objects.active = hulls[0]
bpy.ops.wm.obj_export(
    filepath=str(COLLISION), export_selected_objects=True, apply_modifiers=True,
    forward_axis="Y", up_axis="Z", export_materials=False,
)
print(f"basket: {len(meshes)} visual meshes, {len(hulls)} collision hulls")
