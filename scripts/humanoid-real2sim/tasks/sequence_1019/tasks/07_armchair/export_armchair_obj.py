"""Export the 1012 grey armchair visual mesh in local Z-up metres with Blender."""

from pathlib import Path

import bpy


HERE = Path(__file__).resolve().parents[6] / "outputs/humanoid-real2sim/2026-10-20/sequence_1019/curated_snapshot/tasks/07_armchair"
SOURCE = Path(
    "/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/"
    "objects/1012/grey_armchair_v2.blend"
)
OUTPUT = HERE / "grey_armchair_visual.obj"

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
print(f"armchair: {len(meshes)} visual meshes -> {OUTPUT}")
