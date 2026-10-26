"""Blender-only rigid 90 degree rotation of both authored crown cylinders."""

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path


def main():
    import bpy
    from mathutils import Matrix, Vector

    p = argparse.ArgumentParser()
    p.add_argument("--source-run", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    a = p.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    old = a.source_run
    new = a.run
    for folder in ["scene", "retarget/baseline"]:
        shutil.copytree(old / folder, new / folder)
    for folder in ["logs", "review"]:
        (new / folder).mkdir()
    # Rebase only task-local dependencies. Other scene assets stay unchanged.
    for file in (new / "scene").glob("*"):
        if file.suffix in [".json", ".xml"]:
            file.write_text(file.read_text().replace(str(old), str(new)))
    original = json.loads((old / "scene/provenance.json").read_text())
    changes = []
    rot = Matrix.Rotation(3.141592653589793 / 2, 4, "Z")
    center = Vector((0, 0, 0.110))
    for record in original["meshes"]:
        if not any("cylindrical_grasp_70mm" in x for x in record["prim_paths"]):
            continue
        path = new / "scene" / Path(record["path"]).name
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete(use_global=False)
        bpy.ops.wm.obj_import(filepath=str(path), forward_axis="Y", up_axis="Z")
        for ob in bpy.context.selected_objects:
            for v in ob.data.vertices:
                v.co = center + rot.to_3x3() @ (v.co - center)
        bpy.ops.wm.obj_export(
            filepath=str(path), export_selected_objects=True, forward_axis="Y", up_axis="Z", export_materials=False
        )
        changes.append(
            {
                "file": str(path),
                "source_sha256": hashlib.sha256((old / "scene" / path.name).read_bytes()).hexdigest(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "prim_paths": record["prim_paths"],
            }
        )
    assert len(changes) == 2
    prov = json.loads((new / "scene/provenance.json").read_text())
    prov.update(
        parent_run=str(old),
        geometry_edit={
            "rotation_degrees": 90,
            "rotation_axis": "hat local Z",
            "pivot_m": [0, 0, 0.110],
            "only_modified_parts": changes,
            "tool": "Blender OBJ import/edit/export",
        },
        cylinder={"center": [0, 0, 0.110], "axis": [0, 1, 0], "diameter_m": 0.07, "length_m": 0.205},
    )
    (new / "scene/provenance.json").write_text(json.dumps(prov, indent=2))
    print(json.dumps(changes), flush=True)


if __name__ == "__main__":
    main()
