"""Export an original NX1 episode's fixed ref-to-world transform, without assets.

Input is a trusted dataset traj.npz, or a ZIP with --member naming traj.npz.
This preserves the original episode frame and never adds viewer display offsets.
"""

import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--member", help="exact traj.npz member inside source ZIP")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source
    if args.member:
        with zipfile.ZipFile(source) as archive:
            source = io.BytesIO(archive.read(args.member))
    # These original recordings store joint names as object arrays.
    # Use only the trusted user-provided dataset, then export pickle-free arrays.
    with np.load(source, allow_pickle=True) as data:
        q = np.asarray(data["qpos"], dtype=float)
        ref = np.asarray(data["qpos_ref"], dtype=float)
        human = np.asarray(data["axis_skel_pos"], dtype=float)
        names = np.asarray(data["axis_skel_joints"], dtype=str)
        valid = np.asarray(data["axis_skel_valid"], dtype=bool)
        fps = float(data["sample_hz"])
        times = {
            k: np.asarray(data[k]).copy() for k in ("t_sim", "t_wall") if k in data
        }
    if q.ndim != 2 or ref.ndim != 2 or len(q) == 0 or len(q) != len(ref):
        raise ValueError("invalid paired root arrays")
    if q.shape[1] < 7 or ref.shape[1] < 7:
        raise ValueError("root needs xyz + quaternion wxyz")
    if human.shape != (len(q), len(names), 3) or valid.shape != (len(q),):
        raise ValueError("human frames, names, and validity do not match")
    if not np.isfinite(q[:, :7]).all() or not np.isfinite(ref[:, :7]).all():
        raise ValueError("nonfinite root pose")
    if not np.isfinite(human[valid]).all() or not np.isfinite(fps) or fps <= 0:
        raise ValueError("invalid human positions or sampling rate")
    world_rot = Rotation.from_quat(q[:, [4, 5, 6, 3]])
    ref_rot = Rotation.from_quat(ref[:, [4, 5, 6, 3]])
    rotation = world_rot[0] * ref_rot[0].inv()
    translation = q[0, :3] - rotation.apply(ref[0, :3])
    matrix = np.eye(4)
    matrix[:3, :3] = rotation.as_matrix()
    matrix[:3, 3] = translation
    positions = human @ rotation.as_matrix().T + translation
    root_error = rotation.apply(ref[:, :3]) + translation - q[:, :3]
    angle_error = ((rotation * ref_rot).inv() * world_rot).magnitude()
    report = {
        "source": str(args.source),
        "member": args.member,
        "frames": len(q),
        "fps": fps,
        "root_quaternion_order": "wxyz",
        "formula": "T_world_from_ref = T_world_root[0] @ inverse(T_ref_root[0])",
        "T_world_from_ref": matrix.tolist(),
        "rotation_xyzw": rotation.as_quat().tolist(),
        "translation_m": translation.tolist(),
        "root_position_rms_m": float(np.sqrt(np.mean(root_error**2))),
        "root_position_max_norm_m": float(np.linalg.norm(root_error, axis=1).max()),
        "root_rotation_max_error_rad": float(angle_error.max()),
        "human_valid_frames": int(valid.sum()),
        "display_offset_applied": False,
        "scope": "Recovered from recorded root pair; review residuals before reuse. Not independent test calibration.",
    }
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        args.output / "human_world.npz",
        positions_w=positions,
        joint_names=names,
        valid=valid,
        fps=fps,
        source_frame=np.arange(len(q)),
        T_world_from_ref=matrix,
        **times,
    )
    (args.output / "ref_to_world.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
