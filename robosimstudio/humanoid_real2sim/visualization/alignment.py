"""Fail-fast source-time checks for kinematic human overlays."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def validate_human_alignment(
    overlay: Path,
    *,
    frames: int,
    fps: float,
    source_offset_s: float,
    reference: Path | None = None,
    source_sync: Path | None = None,
) -> dict[str, float]:
    """Check the overlay against a task-local human trace and recording clock."""
    with np.load(overlay, allow_pickle=False) as data:
        positions = np.asarray(data["positions_w"], dtype=np.float64)
        overlay_fps = float(data["fps"])
        source_range = np.asarray(data["source_range_s"], dtype=np.float64) if "source_range_s" in data else None
    if positions.ndim != 3 or positions.shape[0] != frames or positions.shape[2] != 3:
        raise ValueError("human overlay shape does not match retarget motion")
    if not np.isclose(overlay_fps, fps) or not np.isfinite(positions).all():
        raise ValueError("human overlay FPS or positions are invalid")
    if source_range is not None and not np.allclose(
        source_range, [source_offset_s, source_offset_s + frames / fps], atol=0.5 / fps
    ):
        raise ValueError("human overlay source_range_s does not match --source-offset-s")
    result = {"frames": float(frames), "fps": float(fps), "source_offset_s": float(source_offset_s)}
    if reference is not None:
        with np.load(reference, allow_pickle=False) as data:
            key = "keypoints_w" if "keypoints_w" in data else "positions_w"
            expected = np.asarray(data[key], dtype=np.float64)
            reference_fps = float(data["fps"])
        if expected.shape != positions.shape or not np.isclose(reference_fps, fps):
            raise ValueError("task-local human reference shape or FPS does not match overlay")
        error = float(np.max(np.abs(expected - positions)))
        if not np.isfinite(error) or error > 1e-5:
            raise ValueError(f"human overlay is from a different source window (max error {error:.6f} m)")
        result["max_reference_error_m"] = error
    if source_sync is not None:
        with np.load(source_sync, allow_pickle=False) as data:
            recording_time = np.asarray(data["recording_time_s"], dtype=np.float64)
        expected_time = source_offset_s + np.arange(frames) / fps
        if recording_time.shape != (frames,) or not np.isfinite(recording_time).all():
            raise ValueError("source_sync recording_time_s is invalid")
        error = float(np.max(np.abs(recording_time - expected_time)))
        if error > 0.5 / fps:
            raise ValueError(f"source_sync is from a different recording window (max error {error:.6f} s)")
        result["max_recording_time_error_s"] = error
    return result
