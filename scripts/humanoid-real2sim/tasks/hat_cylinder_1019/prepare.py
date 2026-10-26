"""Export the original authored cylindrical hat and isolate the 1019 hat window."""

import argparse
import hashlib
import json
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

OLD = Path("outputs/humanoid-real2sim/2026-10-20/sequence_1019/runs/21-08-52.042738+0800").resolve()
ASSET = Path(
    "/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/objects/1012/black_ribbon_hat_grasped.usd"
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    a = p.parse_args()
    r = a.run.resolve()
    sys.path.insert(0, "/home/galbot/workspace/ScaleTrack/retarget3/src")
    import retarget3.runtime  # noqa: F401
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

    sys.path.insert(0, str(Path("scripts/humanoid-real2sim/tasks/sequence_1019/tasks/03_04_05_joint").resolve()))
    from build_scene import export_meshes

    for sub in ["scene", "retarget/baseline", "review", "logs"]:
        (r / sub).mkdir(parents=True, exist_ok=True)
    scene = r / "scene"
    src = OLD / "retarget/candidates/unified_diagnostic_v3"
    out = r / "retarget/baseline"
    begin, end = 400, 1151  # 35.00--50.00 s, including approach and withdrawal
    for name in ["motion.npz", "human_motion_world.npz", "source_sync.npz"]:
        z = np.load(src / name)
        data = {k: (z[k][begin:end] if z[k].ndim and len(z[k]) == 5104 else z[k]) for k in z.files}
        np.savez_compressed(out / name, **data)
    shutil.copy2(src / "motion_order.json", out / "motion_order.json")
    manifest = json.loads((src / "scene_tracks/tracks.json").read_text())
    (out / "scene_tracks").mkdir(exist_ok=True)
    for item in manifest["objects"] + manifest.get("joints", []):
        z = np.load(src / "scene_tracks" / item["path"])
        data = {k: (z[k][begin:end] if z[k].ndim and len(z[k]) == 5104 else z[k]) for k in z.files}
        np.savez_compressed(out / "scene_tracks" / item["path"], **data)
    (out / "scene_tracks/tracks.json").write_text(json.dumps(manifest, indent=2))
    config = json.loads((OLD / "scene/unified/nine_track_binding_v4_red_berry.json").read_text())
    tree = ET.parse(config["robot_and_early_props"])
    root = tree.getroot()
    body = root.find(".//body[@name='black_ribbon_hat']")
    assert body is not None
    for el in list(body):
        if el.tag == "geom":
            body.remove(el)
    asset = root.find("asset")
    stage = Usd.Stage.Open(str(ASSET))
    records = []
    for i, prim in enumerate(stage.Traverse()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        im = UsdGeom.Imageable(prim)
        visual = im.ComputeVisibility() != "invisible" and im.ComputePurpose() not in ("guide", "proxy")
        collision = prim.HasAPI(UsdPhysics.CollisionAPI) and not visual
        if not (visual or collision):
            continue
        name = "cylinder_hat_" + str(i)
        path = scene / (name + ".obj")
        rec = export_meshes(ASSET, path, included_prefix=str(prim.GetPath()), include_hidden=True)
        color = [0.025, 0.025, 0.025]
        mat = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
        if mat:
            for sp in mat.GetPrim().GetChildren():
                sh = UsdShade.Shader(sp)
                if sh and sh.GetInput("diffuseColor") and sh.GetInput("diffuseColor").Get() is not None:
                    color = list(map(float, sh.GetInput("diffuseColor").Get()))
        ET.SubElement(asset, "mesh", name=name, file=str(path))
        ET.SubElement(
            body,
            "geom",
            name=name,
            type="mesh",
            mesh=name,
            group="2" if visual else "3",
            contype="0" if visual else "1",
            conaffinity="0" if visual else "1",
            mass="0" if visual else ".001",
            rgba=" ".join(map(str, [*color, 1 if visual else 0])),
        )
        records.append(dict(**rec, visual=visual, collision=collision))
    dest = scene / "robot_props_cylinder_hat.xml"
    tree.write(dest, encoding="unicode")
    config["robot_and_early_props"] = str(dest)
    config["status"] = "hat_cylinder_stage3_2_candidate"
    (scene / "config.json").write_text(json.dumps(config, indent=2))
    shutil.copy2(ASSET, scene / ASSET.name)
    (scene / "provenance.json").write_text(
        json.dumps(
            {
                "source": str(ASSET),
                "sha256": hashlib.sha256(ASSET.read_bytes()).hexdigest(),
                "source_motion": str(src / "motion.npz"),
                "source_interval_s": [35, 50],
                "source_frame_slice": [begin, end],
                "cylinder": {"center": [0, 0, 0.110], "axis": [1, 0, 0], "diameter_m": 0.07, "length_m": 0.205},
                "meshes": records,
            },
            indent=2,
        )
    )
    print(r, flush=True)


if __name__ == "__main__":
    main()
