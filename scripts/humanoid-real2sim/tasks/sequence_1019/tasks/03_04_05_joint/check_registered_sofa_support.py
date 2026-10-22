"""Measure hat support on a heightfield sampled from the registered sofa surface."""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def main():
    import mujoco
    import trimesh

    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    a = p.parse_args()
    r = a.run
    base = r / "scene/03_04_05"
    m = trimesh.load(base / "sofa_world.obj", force="mesh")
    T = np.array(json.loads((base / "early_room_basis_comparison.json").read_text())["transforms"]["curated_composed"])
    m.apply_transform(T)
    center = np.array([2.005, 0.82])
    grid = 61
    half = 0.2
    x, y = np.meshgrid(np.linspace(-half, half, grid), np.linspace(-half, half, grid))
    orig = np.c_[x.ravel() + center[0], y.ravel() + center[1], np.full(grid * grid, 2.0)]
    locations, rays, _ = m.ray.intersects_location(orig, np.tile([0, 0, -1], (len(orig), 1)))
    heights = np.full(len(orig), -np.inf)
    np.maximum.at(heights, rays, locations[:, 2])
    assert np.isfinite(heights).all()
    lo = heights.min()
    span = heights.max() - lo
    file = base / "registered_sofa_hat_patch.bin"
    with file.open("wb") as f:
        np.array([grid, grid], dtype=np.int32).tofile(f)
        heights.astype(np.float32).tofile(f)
    tree = ET.parse(base / "hat_rack_physics_early_registered.xml")
    root = tree.getroot()
    world = root.find("worldbody")
    for child in list(world):
        if child.tag == "body" and child.get("name") != "black_ribbon_hat":
            world.remove(child)
    ET.SubElement(
        root.find("asset"),
        "hfield",
        name="registered_sofa_top",
        file=str(file.resolve()),
        size=f"{half} {half} {span} .01",
    )
    ET.SubElement(
        world,
        "geom",
        name="registered_sofa_top",
        type="hfield",
        hfield="registered_sofa_top",
        pos=f"{center[0]} {center[1]} {lo}",
        friction=".8 .005 .001",
    )
    path = base / "hat_registered_sofa_physics.xml"
    tree.write(path, encoding="unicode")
    model = mujoco.MjModel.from_xml_path(str(path))
    d = mujoco.MjData(model)
    z = np.load(r / "retarget/candidates/03_04_05_joint_v24/black_ribbon_hat.npz")
    d.qpos[:3] = z["position_w"][0]
    d.qpos[3:7] = z["quaternion_xyzw"][0][[3, 0, 1, 2]]
    mujoco.mj_forward(model, d)
    initial = [float(c.dist) for c in d.contact]
    before = d.qpos.copy()
    samples = []
    for k in range(5000):
        mujoco.mj_step(model, d)
        if k >= 2500 and k % 50 == 0:
            samples.append(d.qpos.copy())
    samples = np.array(samples)
    report = {
        "status": "physical_surface_patch_diagnostic",
        "sofa_transform": T.tolist(),
        "patch_grid_spacing_m": 2 * half / (grid - 1),
        "patch_height_range_m": [float(lo), float(lo + span)],
        "initial_hat_qpos": before.tolist(),
        "initial_contact_distances_m": initial,
        "settled_hat_qpos": d.qpos.tolist(),
        "settling_translation_m": (d.qpos[:3] - before[:3]).tolist(),
        "last5s_max_translation_m": float(np.linalg.norm(samples[:, :3] - samples[0, :3], axis=1).max()),
        "limitation": "Rigid upper-surface heightfield approximation of actual sofa triangles; not deformable upholstery and not full sofa collision geometry.",
    }
    (r / "review/scene/03_04_05_registered_sofa_support.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report)


if __name__ == "__main__":
    main()
