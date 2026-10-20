"""Blender background conversion of the inherited room USD into RSS world XYZ.

Blender's USD→OBJ bridge writes (x+0.04, z, -y-0.17) for this room.  The
offset comes from its imported root transform; the axis permutation comes
from OBJ export.  Undo both, using the already reviewed coffee-table OBJ as
the coordinate reference.
"""

from pathlib import Path
from robosimstudio.humanoid_real2sim.paths import example_dir

import bpy


HERE = example_dir(__file__)
SOURCE = Path(
    "/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/scenes/"
    "tianze_sim_data/IKEA_LivingRoom_Mesh_1012/"
    "IKEA_LivingRoom_Mesh_1012_drawer_renderable_v4.usd"
)
OUTPUT = HERE / "assets/ikea_1012_room.obj"
REFERENCE = HERE.parent / "push_bin_1019/assets/coffee_table_visual.obj"

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)
bpy.ops.wm.usd_import(filepath=str(SOURCE), import_materials=False, import_cameras=False, import_lights=False)
meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
if not meshes:
    raise RuntimeError("USD import contained no meshes")
bpy.ops.object.select_all(action="DESELECT")
for obj in meshes:
    obj.select_set(True)
bpy.context.view_layer.objects.active = meshes[0]
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
bpy.ops.wm.obj_export(filepath=str(OUTPUT), export_selected_objects=True, apply_modifiers=True)


def coffee_vertices(path):
    inside = False
    points = []
    for line in path.open():
        if line.startswith("o "):
            inside = "part_0000_CoffeeTable_rounded_top" in line
        elif inside and line.startswith("v "):
            points.append(tuple(map(float, line.split()[1:4])))
    return points


before = coffee_vertices(OUTPUT)
reference = coffee_vertices(REFERENCE)
if len(before) != len(reference) or len(before) != 256:
    raise RuntimeError("cannot calibrate room OBJ against reviewed coffee table")
offset_x = sum(a[0] - b[0] for a, b in zip(before, reference)) / len(before)
offset_y = sum(-a[2] - b[1] for a, b in zip(before, reference)) / len(before)
error = max(
    max(abs(a[0] - offset_x - b[0]), abs(-a[2] - offset_y - b[1]), abs(a[1] - b[2]))
    for a, b in zip(before, reference)
)
if error > 2e-5:
    raise RuntimeError(f"USD→OBJ transform is not rigid or table vertex order changed: {error:.6f} m")

aligned = OUTPUT.with_suffix(".aligned.obj")
with OUTPUT.open() as source, aligned.open("w") as dest:
    for line in source:
        if line.startswith("v "):
            x, y, z = map(float, line.split()[1:4])
            line = f"v {x-offset_x:.9f} {-z-offset_y:.9f} {y:.9f}\n"
        elif line.startswith("vn "):
            x, y, z = map(float, line.split()[1:4])
            line = f"vn {x:.9f} {-z:.9f} {y:.9f}\n"
        dest.write(line)
aligned.replace(OUTPUT)
print(f"exported {len(meshes)} meshes to {OUTPUT}; alignment offsets=({offset_x:.6f}, {offset_y:.6f}) m, coffee residual={error:.7f} m")
