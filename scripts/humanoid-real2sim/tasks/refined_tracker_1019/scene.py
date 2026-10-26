"""Load archived 1019 scenes for physical tracker review, with frame-zero prop poses."""

import importlib.util
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
TASKS = ROOT / "scripts/humanoid-real2sim/tasks"
SNAP = ROOT / "outputs/humanoid-real2sim/2026-10-20"


def load(path):
    spec = importlib.util.spec_from_file_location("review_scene_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def track(path):
    with np.load(path, allow_pickle=False) as z:
        quat = z["quaternion_xyzw"] if "quaternion_xyzw" in z else z["quaternion_wxyz"][:, [1, 2, 3, 0]]
        return {"position_w": z["position_w"], "quaternion_xyzw": quat, "fps": z["fps"]}


def build_sim(asset, motion):
    package = Path(os.environ["RSS_REVIEW_PACKAGE"])
    work = Path(os.environ["RSS_REVIEW_SCENE_OUT"])
    work.mkdir(parents=True, exist_ok=True)
    name = package.name
    objects = {}
    joints = {}
    if name in ("hat-to-stand-kinematic-v1", "drawer-push-kinematic-v1"):
        config = json.loads((package / "scene/config.json").read_text())
        # asset is the package robot+props MJCF with tracker position actuators applied.
        config["robot_and_early_props"] = str(asset)
        path = work / "config.json"
        path.write_text(json.dumps(config, indent=2))
        os.environ["RSS_SEQUENCE_SCENE_CONFIG"] = str(path)
        module = load(package / "scene/rss_scene.py")
        manifest = package / "metadata/scene_tracks/tracks.json"
        info = json.loads(manifest.read_text())
        objects = {e["name"]: track(manifest.parent / e["path"]) for e in info["objects"]}
        joints = {e["name"]: float(np.load(manifest.parent / e["path"])["position"][0]) for e in info.get("joints", [])}
    elif name.startswith("push-armchair") or name.startswith("brace-dining"):
        module = load(TASKS / "sequence_1019/tasks/07_armchair/scene_stage4.py")
        objects["grey_armchair"] = track(package / "metadata/object_trajectories/grey_armchair.npz")
        np.savez(work / "object_estimate.npz", **objects["grey_armchair"])
        os.environ["RSS_ARMCHAIR_STAGE4_CANDIDATE"] = str(work)
        os.environ["RSS_ARMCHAIR_STAGE4_ASSETS"] = str(
            SNAP / "sequence_1019__armchair_stage4/runs/20-06-17.365091+0800/assets_v3"
        )
    elif name.startswith("lined-rect"):
        module = load(TASKS / "sequence_1019/tasks/09_lined_rect_basket/scene_stage4.py")
        objects["lined_rect_basket"] = track(package / "lined_rect_basket_trajectory.npz")
        np.savez(work / "object_estimate.npz", **objects["lined_rect_basket"])
        provenance = json.loads((package / "metadata/provenance.json").read_text())
        visual = Path(provenance["source_motion"]).parent / "basket_visual.obj"
        target = work / "basket_visual.obj"
        if not target.exists():
            target.symlink_to(visual)
        os.environ["RSS_BASKET_STAGE4_CANDIDATE"] = str(work)
        os.environ["RSS_BASKET_SHOW_ROOM"] = "1"
    elif name.startswith("round-basket"):
        module = load(TASKS / "sequence_1019/tasks/06_round_basket/scene_stage4.py")
        objects["round_laundry_basket"] = track(package / "round_laundry_basket_trajectory.npz")
        np.savez(work / "object_estimate.npz", **objects["round_laundry_basket"])
    elif name.startswith("brush-sweep"):
        module = load(TASKS / "brush_sweep_1019/scene.py")
        module.ASSETS = SNAP / "brush_pick_1012/curated_snapshot/assets"
        module.BIN_ASSETS = SNAP / "push_bin_1019/curated_snapshot/assets"
        objects = {p.stem: track(p) for p in (package / "metadata/object_trajectories").glob("*.npz")}
    elif name.startswith("push-bin"):
        module = load(TASKS / "push_bin_1019/scene.py")
        objects["white_waste_bin"] = track(package / "white_waste_bin_trajectory.npz")
    else:
        raise ValueError(f"No reviewed physical scene mapping for {name}")
    if name.startswith("round-basket"):
        hull_root = SNAP / "sequence_1019__armchair_stage4/runs/20-06-17.365091+0800/assets_v3"
        hull_info = json.loads((hull_root / "collision_manifest.json").read_text())
        sim = module.build_sim(
            asset,
            motion,
            basket_mass_kg=1.6,
            basket_com_z_m=0.08,
            basket_track_path=work / "object_estimate.npz",
            table_collision_paths=tuple(hull_root / p for p in hull_info["objects"]["table"]["hulls"]),
        )
    else:
        sim = module.build_sim(asset, motion)
    q = sim.pipeline.snapshot.joint_q.copy()
    starts = sim.model.joint_q_start.numpy()
    children = sim.model.joint_child.numpy()
    body_labels = [str(x).split("/")[-1] for x in sim.model.body_label]
    labels = [str(x).split("/")[-1] for x in sim.model.joint_label]
    assigned = {}
    for obj, values in objects.items():
        matches = [i for i, label in enumerate(labels) if label == obj or body_labels[int(children[i])] == obj]
        matches = [i for i in matches if starts[i + 1] - starts[i] == 7]
        if len(matches) != 1:
            raise ValueError(f"{obj}: expected one free joint, found {matches}; labels={labels}")
        i = int(starts[matches[0]])
        q[i : i + 3] = values["position_w"][0]
        q[i + 3 : i + 7] = values["quaternion_xyzw"][0]
        assigned[obj] = q[i : i + 7].tolist()
    for joint, value in joints.items():
        q[int(starts[labels.index(joint)])] = value
    sim._start_joint_q = q
    sim.reset()
    (work / "scene_audit.json").write_text(
        json.dumps(
            {
                "package": str(package),
                "source_scene": module.__file__,
                "physical_bodies": body_labels,
                "initial_objects": assigned,
                "initial_joints": joints,
                "shape_count": sim.model.shape_count,
                "checkpoint": os.environ.get("RSS_REVIEW_CHECKPOINT"),
                "status": "scene built; object poses initialized once; interaction success unverified",
            },
            indent=2,
        )
    )
    return sim
