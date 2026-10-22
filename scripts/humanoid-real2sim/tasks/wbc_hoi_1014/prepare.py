"""Adapt two original NX1 episodes to the established Stage 3 review contract."""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    args = ap.parse_args()
    run = args.run.resolve()
    urdf = ET.parse(
        run / "nx1_description/urdf/v1/v1_0/galbot_nx1_v1_0_1.urdf"
    ).getroot()
    for node in urdf.iter("mesh"):
        node.set(
            "filename",
            str(
                run
                / "nx1_description"
                / node.get("filename").split("galbot_nx1_description/")[1]
            ),
        )
    ET.SubElement(
        ET.SubElement(urdf, "mujoco"),
        "compiler",
        {"discardvisual": "false", "fusestatic": "false", "balanceinertia": "true"},
    )
    ET.ElementTree(urdf).write(run / "nx1.urdf")
    imported = mujoco.MjModel.from_xml_path(str(run / "nx1.urdf"))
    mujoco.mj_saveLastXML(str(run / "nx1.xml"), imported)
    tree = ET.parse(run / "nx1.xml")
    root = tree.getroot()
    base = root.find("./worldbody/body")
    # URDF's zero-mass base is only a frame; the fixed pelvis has the physical inertia.
    pelvis = base.find("body[@name='pelvis_link']")
    if pelvis is not None:
        base.remove(pelvis)
        root.find("worldbody").remove(base)
        root.find("worldbody").append(pelvis)
        base = pelvis
    ET.SubElement(base, "freejoint", {"name": "root"})
    for geom in root.iter("geom"):
        geom.set("contype", "0")
        geom.set("conaffinity", "0")
    tree.write(run / "nx1_free.xml")
    model = mujoco.MjModel.from_xml_path(str(run / "nx1_free.xml"))
    names = tuple(
        model.joint(i).name for i in range(model.njnt) if model.jnt_type[i] != 0
    )
    parents = [-1, 0, 1, 2, 3, 4, 3, 6, 7, 8, 3, 10, 11, 12, 0, 14, 15, 0, 17, 18]
    edges = np.array([(p, i) for i, p in enumerate(parents) if p >= 0])
    motions = []
    humans = []
    segments = []
    audit = []
    start = 0
    for idx, label in enumerate(("01_1014", "02_1014")):
        folder = run / label
        with np.load(folder / "traj.npz", allow_pickle=True) as z:
            meta = json.loads(str(z["meta"]))
            q = z["qpos"].astype(float)
            ref = z["qpos_ref"].astype(float)
            joints = meta["robot_model"]["joint_order"]
            human = z["axis_skel_pos"].astype(float)
            assert np.all(z["axis_skel_valid"]) and float(z["sample_hz"]) == 50
            # Original skeleton is in the qpos_ref world; apply one recorded spawn transform.
            rot = (
                Rotation.from_quat(q[0, [4, 5, 6, 3]])
                * Rotation.from_quat(ref[0, [4, 5, 6, 3]]).inv()
            )
            trans = q[0, :3] - rot.apply(ref[0, :3])
            shift = np.array([12.0 * idx, 0, 0])
            human = rot.apply(human.reshape(-1, 3)).reshape(human.shape) + trans + shift
            q[:, :3] += shift
            joint_pos = q[:, [7 + joints.index(n) for n in names]]
            motion = {
                "joint_pos": joint_pos,
                "body_pos_w": q[:, :3, None].transpose(0, 2, 1),
                "body_quat_w": q[:, None, 3:7],
                "robot_joint_names": np.array(names),
                "robot_body_names": np.array([base.get("name")]),
                "fps": np.array(50),
            }
            np.savez_compressed(folder / "motion.npz", **motion)
            np.savez_compressed(
                folder / "human.npz",
                positions_w=human,
                joint_names=z["axis_skel_joints"].astype(str),
                edges=edges,
                fps=50,
                source_range_s=[0, len(q) / 50],
            )
            object_pose = z["object_pose"][:, 0].astype(float)
            object_pose[:, :3] += shift
            np.savez_compressed(
                folder / "object_track.npz",
                position_w=object_pose[:, :3],
                quaternion_xyzw=object_pose[:, [4, 5, 6, 3]],
                fps=50,
            )
            (folder / "object_config.json").write_text(str(z["object_config_json"]))
            (folder / "meta.json").write_text(json.dumps(meta, indent=2))
            motions.append(motion)
            humans.append(human)
            segments.append(
                {
                    "id": "Sample " + label + " / " + meta["episode_id"],
                    "start": start,
                    "stop": start + len(q),
                }
            )
            audit.append(
                {
                    "sample": label,
                    "episode": meta["episode_id"],
                    "frames": len(q),
                    "fps": 50,
                    "source": "teleop_replay",
                    "robot": meta["robot_model"],
                    "human_alignment_rotation_xyzw": rot.as_quat().tolist(),
                    "human_alignment_translation": trans.tolist(),
                    "display_translation": shift.tolist(),
                    "root_ref_actual_rms": float(
                        np.sqrt(
                            np.mean(
                                (rot.apply(ref[:, :3]) + trans - (q[:, :3] - shift))
                                ** 2
                            )
                        )
                    ),
                    "qpos_ref_joint_max_difference": float(
                        np.max(np.abs(ref[:, 7:39] - q[:, 7:39]))
                    ),
                }
            )
            start += len(q)
    combined = {
        k: np.concatenate([m[k] for m in motions])
        if k in ("joint_pos", "body_pos_w", "body_quat_w")
        else motions[0][k]
        for k in motions[0]
    }
    np.savez_compressed(run / "motion.npz", **combined)
    np.savez_compressed(
        run / "human.npz",
        positions_w=np.concatenate(humans),
        joint_names=np.array(
            [
                "Hips",
                "Spine",
                "Spine1",
                "Spine2",
                "Neck",
                "Head",
                "RightShoulder",
                "RightArm",
                "RightForeArm",
                "RightHand",
                "LeftShoulder",
                "LeftArm",
                "LeftForeArm",
                "LeftHand",
                "RightUpLeg",
                "RightLeg",
                "RightFoot",
                "LeftUpLeg",
                "LeftLeg",
                "LeftFoot",
            ]
        ),
        edges=edges,
        fps=50,
        source_range_s=[0, start / 50],
    )
    (run / "segments.json").write_text(json.dumps(segments, indent=2))
    (run / "audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
