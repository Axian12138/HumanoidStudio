"""Subprocess seam between RSS (Python 3.12) and ScaleBridge (Python 3.11)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


class BFMTracker:
    def __init__(self, *, python: Path, scalebridge: Path, checkpoint: Path, metadata: Path):
        self.metadata = json.loads(Path(metadata).read_text())
        self.process = subprocess.Popen(
            [
                str(python),
                str(Path(__file__).with_name("worker.py")),
                "--scalebridge",
                str(scalebridge),
                "--checkpoint",
                str(checkpoint),
                "--metadata",
                str(metadata),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
        )
        try:
            line = self._response()
            if not line.startswith("@ready "):
                raise RuntimeError(f"tracker worker did not become ready: {line}")
            declared = json.loads(line.removeprefix("@ready "))
            if (
                declared["joint_names"] != self.metadata["joint_names"]
                or declared["body_names"] != self.metadata["selected_body_names"]
            ):
                raise RuntimeError("worker body or joint order differs from metadata")
        except BaseException:
            self.close()
            raise

    def _response(self) -> str:
        assert self.process.stdout is not None
        for line in self.process.stdout:
            if line.startswith("@"):
                if line.startswith("@error "):
                    error = json.loads(line.removeprefix("@error "))
                    raise RuntimeError(f"tracker inference failed: {error['type']}: {error['message']}")
                return line.rstrip("\n")
        raise RuntimeError(f"tracker worker exited with code {self.process.poll()}")

    def infer(self, observation: dict[str, Any], mode: int) -> tuple[np.ndarray, np.ndarray]:
        if self.process.poll() is not None:
            raise RuntimeError(f"tracker worker exited with code {self.process.returncode}")
        assert self.process.stdin is not None
        payload = {
            "command": "infer",
            "mode": int(mode),
            "observation": {key: np.asarray(value).tolist() for key, value in observation.items()},
        }
        self.process.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
        self.process.stdin.flush()
        response = self._response()
        if not response.startswith("@result "):
            raise RuntimeError(f"unexpected tracker response: {response}")
        answer = json.loads(response.removeprefix("@result "))
        target = np.asarray(answer["target"], dtype=np.float32)
        action = np.asarray(answer["action"], dtype=np.float32)
        count = len(self.metadata["action_names"])
        if (
            target.shape != (count,)
            or action.shape != (count,)
            or not (np.isfinite(target).all() and np.isfinite(action).all())
        ):
            raise RuntimeError("tracker returned an invalid action")
        return target, action

    def close(self) -> None:
        if self.process.poll() is not None:
            return
        try:
            if self.process.stdin is not None:
                self.process.stdin.write('{"command":"close"}\n')
                self.process.stdin.flush()
            self.process.wait(timeout=3)
        except (BrokenPipeError, subprocess.TimeoutExpired):
            self.process.terminate()
            self.process.wait(timeout=3)
