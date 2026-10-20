"""Hold the aligned contact pose and measure physical Dex3/brush contacts."""

import argparse
import json
import sys
from pathlib import Path
from robosimstudio.humanoid_real2sim.paths import SCRIPTS, example_dir

import newton
import numpy as np

sys.path.insert(0, str(SCRIPTS))
from robosimstudio.humanoid_real2sim.track import launcher

from robosimstudio.humanoid_real2sim.track.timeline import ReferenceMotion
from robosimstudio.scene.entity import label_index

ROOT = example_dir(__file__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand-kp", type=float, default=2.0)
    parser.add_argument("--hand-kv", type=float, default=0.1)
    parser.add_argument("--motion", type=Path, default=ROOT / "retarget/palm_waist_v3/motion.npz")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    metadata = json.loads(launcher.DEFAULT_METADATA.read_text())
    motion = ReferenceMotion.load(
        args.motion,
        joint_names=tuple(metadata["joint_names"]),
        body_names=tuple(metadata["selected_body_names"]),
        motion_order=launcher.DEFAULT_ORDER,
    )
    temporary, asset = launcher._position_asset(
        launcher.DEFAULT_ROBOT, metadata, hand_kp=args.hand_kp, hand_kv=args.hand_kv,
    )
    try:
        sim = launcher._load_scene_file(SCRIPTS / "tasks/brush_pick_1012/scene.py", asset, motion)
        names = list(sim.model.joint_label)
        starts = sim.model.joint_q_start.numpy()
        dof_starts = sim.model.joint_qd_start.numpy()
        coords = {name: int(starts[label_index(names, name)]) for name in motion.all_joint_names}
        dofs = {name: int(dof_starts[label_index(names, name)]) for name in motion.all_joint_names}
        q = np.asarray(sim._start_joint_q, dtype=np.float32).copy()
        root = int(starts[label_index(names, "floating_base")])
        q[root:root + 3] = motion.body_pos_w[560, 0]
        q[root + 3:root + 7] = motion.body_quat_w[560, 0, [1, 2, 3, 0]]
        for name, value in zip(motion.all_joint_names, motion.all_joint_pos[560], strict=True):
            q[coords[name]] = value
        for name in coords:
            if name.startswith("right_hand_"):
                q[coords[name]] = motion.all_joint_pos[500, motion.all_joint_names.index(name)]
        sim._start_joint_q = q
        sim.reset()
        body_names = list(sim.model.body_label)
        brush = label_index(body_names, "green_hand_brush")
        wrist = label_index(body_names, "right_wrist_yaw_link")
        pelvis = label_index(body_names, "pelvis")
        shape_body = sim.model.shape_body.numpy()
        brush_shapes = np.flatnonzero(shape_body == brush)
        groups = {}
        for part in ("right_hand_thumb_0_link", "right_hand_thumb_1_link", "right_hand_thumb_2_link",
                     "right_hand_middle_0_link", "right_hand_middle_1_link",
                     "right_hand_index_0_link", "right_hand_index_1_link"):
            body = label_index(body_names, part)
            groups[part] = np.flatnonzero(shape_body == body)
        probe = sim._make_contact_probe()
        if probe is None:
            raise RuntimeError("RSS scene did not provide a contact probe")
        samples = []
        for step in range(80):
            target = sim.pipeline.snapshot.joint_target_q.astype(np.float32).copy()
            for name in coords:
                index = motion.all_joint_names.index(name)
                if name.startswith("right_hand_"):
                    finger_frame = int(round(500 + min(step / 40, 1) * 60))
                    target[coords[name]] = motion.all_joint_pos[finger_frame, index]
                else:
                    target[coords[name]] = motion.all_joint_pos[560, index]
            sim.control.joint_target_q.assign(target)
            sim.step_teleop()
            probe.update(sim.pipeline.step_count)
            if step % 5 == 0 or step == 79:
                forces = {name: float(np.linalg.norm(probe.pairwise_force(shapes, brush_shapes)))
                          for name, shapes in groups.items()}
                a, b = probe._shape_pairs()
                contacts = []
                for shape_a, shape_b, impulse in zip(a, b, probe.impulses, strict=True):
                    if shape_a in brush_shapes or shape_b in brush_shapes:
                        other = shape_b if shape_a in brush_shapes else shape_a
                        contacts.append({"other_body": body_names[int(shape_body[other])],
                                         "impulse_Ns": float(np.linalg.norm(impulse))})
                samples.append({
                    "step": step,
                    "brush_position": sim.pipeline.snapshot.body_q[brush, :3].astype(float).tolist(),
                    "wrist_position": sim.pipeline.snapshot.body_q[wrist, :3].astype(float).tolist(),
                    "pelvis_position": sim.pipeline.snapshot.body_q[pelvis, :3].astype(float).tolist(),
                    "finger_forces_N": forces,
                    "brush_contacts": contacts,
                    "finger_targets_rad": {name: float(target[coord]) for name, coord in coords.items()
                                            if name.startswith("right_hand_")},
                    "finger_actual_rad": {name: float(sim.pipeline.snapshot.joint_q[coord])
                                          for name, coord in coords.items() if name.startswith("right_hand_")},
                })
            # Isolate grasp mechanics: prescribe the approved whole-body pose,
            # while fingers and the brush remain physically simulated.
            pinned_q = sim.pipeline.snapshot.joint_q.astype(np.float32).copy()
            pinned_qd = sim.pipeline.snapshot.joint_qd.astype(np.float32).copy()
            pinned_q[root:root + 7] = q[root:root + 7]
            root_dof = int(dof_starts[label_index(names, "floating_base")])
            pinned_qd[root_dof:root_dof + 6] = 0
            for name in coords:
                if not name.startswith("right_hand_"):
                    pinned_q[coords[name]] = motion.all_joint_pos[560, motion.all_joint_names.index(name)]
                    pinned_qd[dofs[name]] = 0
            for state in (sim.pipeline.state_0, sim.pipeline.state_1):
                state.joint_q.assign(pinned_q)
                state.joint_qd.assign(pinned_qd)
                newton.eval_fk(sim.model, state.joint_q, state.joint_qd, state)
            sim.pipeline.mark_dirty()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"hand_kp": args.hand_kp, "hand_kv": args.hand_kv,
                                        "brush_shape_count": len(brush_shapes),
                                        "finger_shape_counts": {name: len(shapes) for name, shapes in groups.items()},
                                        "samples": samples}, indent=2) + "\n")
        print(args.out)
    finally:
        temporary.cleanup()


if __name__ == "__main__":
    main()
