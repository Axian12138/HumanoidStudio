"""Slow only the native preview clock; preserve source timestamps and motion FPS."""

import hashlib
import json
import os
import runpy
import time
from pathlib import Path
from threading import Thread
from types import SimpleNamespace

import robosimstudio.humanoid_real2sim.visualization.kinematic as module

rate = float(os.environ.get("REVIEW_RATE", "1"))
if not 0 < rate <= 1:
    raise ValueError("REVIEW_RATE must lie in (0, 1]")
wall = time.monotonic
origin = wall()
module.time = SimpleNamespace(monotonic=lambda: origin + (wall() - origin) * rate)
capture_dir = os.environ.get("REVIEW_CAPTURE_DIR")
if capture_dir:
    destination = Path(capture_dir)
    destination.mkdir(parents=True, exist_ok=False)
    original_init = module.KinematicStudio.__init__

    def initialize_with_capture(self, *args, **kwargs):
        original_init(self, *args, **kwargs)

        def capture():
            from PIL import Image

            last = None
            counter = 0
            time.sleep(2)
            while True:
                clients = self.server.get_clients()
                frame = self.cursor
                if not clients or (last is not None and abs(frame - last) < 25 and frame != self.stop - 1):
                    time.sleep(0.2)
                    continue
                if last == frame:
                    time.sleep(0.2)
                    continue
                client_id = max(clients)
                client = clients[client_id]
                before = self._public_state()
                requested = time.time()
                record = {"client_id": client_id, "request_unix_s": requested, "state_before": before}
                try:
                    pixels = client.get_render(height=720, width=1280, transport_format="jpeg", timeout=8)
                    returned = time.time()
                    after = self._public_state()
                    path = destination / f"{counter:04d}_frame_{before['cursor']:04d}.png"
                    Image.fromarray(pixels).save(path)
                    record.update(
                        return_unix_s=returned,
                        state_after=after,
                        image=path.name,
                        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        scope="Original connected browser client public render API; concurrent playback frame interval explicitly recorded.",
                    )
                    counter += 1
                except Exception as exc:
                    record.update(return_unix_s=time.time(), error=repr(exc), state_after=self._public_state())
                with (destination / "captures.jsonl").open("a") as stream:
                    stream.write(json.dumps(record) + "\n")
                last = frame
                time.sleep(0.2)

        Thread(target=capture, daemon=True).start()

    module.KinematicStudio.__init__ = initialize_with_capture
print(f"Review clock rate={rate}; source FPS, timestamps, and motion data unchanged", flush=True)
runpy.run_path("scripts/humanoid-real2sim/kinematic_viewer.py", run_name="__main__")
