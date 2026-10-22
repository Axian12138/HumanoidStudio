"""Validated, source-aligned object and mechanism tracks for kinematic review."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SceneTrack:
    name: str
    kind: str
    values: np.ndarray
    path: Path


def load_scene_tracks(manifest: Path, *, frames: int, fps: float, source_offset_s: float) -> list[SceneTrack]:
    """Load strict multi-track manifests; timestamps are recording seconds."""
    document = json.loads(manifest.read_text())
    tracks = []
    names = set()
    expected_time = source_offset_s + np.arange(frames) / fps
    for group, kind in (("objects", "free"), ("joints", "scalar")):
        for entry in document.get(group, []):
            name = entry["name"]
            if not isinstance(name, str) or not name or name in names:
                raise ValueError(f"invalid or duplicate scene track name: {name!r}")
            names.add(name)
            path = (manifest.parent / entry["path"]).resolve()
            with np.load(path, allow_pickle=False) as data:
                times = np.asarray(data["source_time_s"], dtype=float)
                if times.shape != (frames,) or not np.allclose(times, expected_time, rtol=0, atol=1e-6):
                    raise ValueError(f"{name}: track source timestamps do not match motion")
                if not np.isclose(float(data["fps"]), fps, rtol=0, atol=1e-6):
                    raise ValueError(f"{name}: track FPS does not match motion")
                if kind == "free":
                    position = np.asarray(data["position_w"], dtype=np.float32)
                    quaternion = np.asarray(data["quaternion_xyzw"], dtype=np.float32)
                    if position.shape != (frames, 3) or quaternion.shape != (frames, 4):
                        raise ValueError(f"{name}: expected Nx3 positions and Nx4 xyzw quaternions")
                    if not np.allclose(np.linalg.norm(quaternion, axis=1), 1, atol=1e-4):
                        raise ValueError(f"{name}: quaternions must have unit norm")
                    values = np.concatenate((position, quaternion), axis=1)
                else:
                    values = np.asarray(data["position"], dtype=np.float32)
                    if values.shape != (frames,):
                        raise ValueError(f"{name}: expected N scalar joint positions")
                    values = values[:, None]
                if not np.isfinite(values).all():
                    raise ValueError(f"{name}: track contains non-finite values")
            tracks.append(SceneTrack(name, kind, values, path))
    if not tracks:
        raise ValueError("scene track manifest is empty")
    return tracks


def bind_scene_tracks(tracks, labels, starts, root_coord, robot_coords):
    """Resolve named coordinates and prevent overlapping or wrong-width writes."""
    from robosimstudio.scene.entity import label_index

    occupied = set(range(root_coord, root_coord + 7)) | set(map(int, robot_coords))
    bindings = []
    for track in tracks:
        candidates = [track.name]
        if track.kind == "free":
            candidates.append(f"{track.name}_free_joint")
        for name in candidates:
            try:
                index = label_index(labels, name)
                break
            except KeyError:
                continue
        else:
            raise ValueError(f"scene joint for track {track.name!r} not found")
        coord = int(starts[index])
        width = track.values.shape[1]
        if int(starts[index + 1]) - coord != width:
            raise ValueError(f"{track.name}: scene joint coordinate width does not match track")
        coordinates = set(range(coord, coord + width))
        if occupied & coordinates:
            raise ValueError(f"{track.name}: track overlaps robot or another scene track")
        occupied.update(coordinates)
        bindings.append((track, coord))
    return bindings
