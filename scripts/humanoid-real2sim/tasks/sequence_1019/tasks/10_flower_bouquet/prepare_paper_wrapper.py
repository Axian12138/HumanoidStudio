"""Version a source-observed tapered paper envelope, preserving the original asset."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    import trimesh

    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    folder = args.run / "scene/10_flower_bouquet"
    source = folder / "registered_assets_measured_support_v3"
    dest = folder / "registered_assets_paper_wrapper_v1"
    dest.mkdir(exist_ok=False)
    for p in source.iterdir():
        if p.name not in {"flower_bouquet.obj", "flower_bouquet.mtl", "flower_bouquet_hull0.obj", "provenance.json"}:
            (dest / p.name).symlink_to(p.resolve())
    sections = 48
    theta = np.arange(sections) * 2 * np.pi / sections
    lower, upper, height, thickness = 0.020, 0.033, 0.360078, 0.0005

    def ring(radius, z):
        return np.c_[radius * np.cos(theta), radius * np.sin(theta), np.full(sections, z)]

    vertices = np.concatenate(
        [ring(lower, 0), ring(upper, height), ring(lower - thickness, 0), ring(upper - thickness, height)]
    )
    faces = []
    for a, b in [(0, 1), (3, 2), (2, 0), (1, 3)]:
        for i in range(sections):
            j = (i + 1) % sections
            faces.extend(
                [
                    [a * sections + i, a * sections + j, b * sections + j],
                    [a * sections + i, b * sections + j, b * sections + i],
                ]
            )
    shell = trimesh.Trimesh(vertices, faces, process=False)
    trimesh.repair.fix_normals(shell)
    raw = (source / "flower_bouquet.obj").read_text()
    offset = sum(line.startswith("v ") for line in raw.splitlines())
    with (dest / "flower_bouquet.obj").open("w") as stream:
        stream.write(raw + "\n")
        for v in shell.vertices:
            stream.write("v " + " ".join(map(str, v)) + "\n")
        stream.write("usemtl PaperWrapperEstimate\n")
        for f in shell.faces:
            stream.write("f " + " ".join(str(int(i) + offset + 1) for i in f) + "\n")
    (dest / "flower_bouquet.mtl").write_text(
        (source / "flower_bouquet.mtl").read_text()
        + "\nnewmtl PaperWrapperEstimate\nKd 0.88 0.89 0.88\nKa 0.1 0.1 0.1\nKs 0.02 0.02 0.02\nNs 5\n"
    )
    hull = trimesh.convex.convex_hull(vertices[: 2 * sections])
    hull.export(dest / "flower_bouquet_hull0.obj")
    profile = {
        "lower_radius_m": lower,
        "upper_radius_m": upper,
        "height_m": height,
        "visual_shell_thickness_m": thickness,
    }
    (dest / "wrapper_profile.json").write_text(json.dumps(profile, indent=2))
    (dest / "provenance.json").write_text(
        json.dumps(
            {
                "status": "source-observed tapered wrapper candidate; multi-view and full hand collision review required",
                "source_assets": str(source),
                "profile": profile,
                "evidence": "review/scene/10_paper_wrapper_independent/measurements.json; calibrated five-frame upper diameter59-66mm and lower40-45mm",
                "limitations": "radii are bounded visual estimates, shell thickness is a rendering assumption, not measured paper physics; collision uses full outer frustum",
                "visual_shell_volume_m3": float(shell.volume),
                "unchanged": "leaf envelope, source-authored masses, measured cabinet/wall poses, all old variants",
            },
            indent=2,
        )
    )
    print(dest)


if __name__ == "__main__":
    main()
