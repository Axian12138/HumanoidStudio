"""Diagnose independent support contact by shape group, without a robot."""

import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    from scene_registered import make_scene

    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--per-shape-contacts", action="store_true")
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    inputs = {}
    for key in ["RSS_FLOWER_ASSETS", "RSS_FLOWER_INITIAL_TRACKS"]:
        if key not in os.environ:
            inputs[key] = {"path": None, "scope": "Scene default; no explicit environment override."}
            continue
        folder = Path(os.environ[key])
        inputs[key] = {
            "path": str(folder.resolve()),
            "files": {
                str(path.relative_to(folder)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(folder.rglob("*"))
                if path.is_file()
            },
        }
    inputs["audit_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    inputs["scene_script_sha256"] = hashlib.sha256(
        Path(__file__).with_name("scene_registered.py").read_bytes()
    ).hexdigest()
    sim = make_scene(room=False).build()
    bodies = sim.model.body_label
    labels = sim.model.shape_label
    sb = sim.model.shape_body.numpy()
    groups = {
        name: [i for i, label in enumerate(labels) if name in label]
        for name in [
            "flower_bouquet_collision",
            "small_wicker_basket_collision",
            "cabinet_side",
            "cabinet_support",
            "room_back_wall",
        ]
        if any(name in label for label in labels)
    }
    records = []
    for i in range(126):
        if i:
            sim.run(2).sync()
        sim.contacts.update()
        records.append(
            {
                "time_s": i * 0.04,
                "flower": sim["flower_bouquet"].pose.tolist(),
                "basket": sim["small_wicker_basket"].pose.tolist(),
                "contacts": {
                    name: bool(sim.contacts.in_contact(groups["flower_bouquet_collision"], ids))
                    for name, ids in groups.items()
                    if name != "flower_bouquet_collision"
                },
                "forces": {
                    name: sim.contacts.pairwise_force(groups["flower_bouquet_collision"], ids).tolist()
                    for name, ids in groups.items()
                    if name != "flower_bouquet_collision"
                },
            }
        )
        if a.per_shape_contacts:
            records[-1]["shape_contacts"] = [
                {
                    "flower_shape_id": flower_id,
                    "flower_shape_label": labels[flower_id],
                    "other_shape_id": other_id,
                    "other_shape_label": labels[other_id],
                    "force": sim.contacts.pairwise_force([flower_id], [other_id]).tolist(),
                }
                for flower_id in groups["flower_bouquet_collision"]
                for name, ids in groups.items()
                if name != "flower_bouquet_collision"
                for other_id in ids
                if sim.contacts.in_contact([flower_id], [other_id])
            ]
    a.out.write_text(
        json.dumps(
            {
                "provenance": inputs,
                "body_labels": bodies,
                "shape_labels": labels,
                "shape_body": sb.tolist(),
                "groups": groups,
                "body_com": sim.model.body_com.numpy().tolist(),
                "body_mass": sim.model.body_mass.numpy().tolist(),
                "records": records,
            },
            indent=2,
        )
    )
    print(a.out)


if __name__ == "__main__":
    main()
