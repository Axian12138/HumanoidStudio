# SPDX-License-Identifier: Apache-2.0
"""Save a RoboSimStudio CPU/GPU tiled render of the hurdle scene."""

import os
from pathlib import Path

from PIL import Image

os.environ.setdefault("ROBOSIMSTUDIO_RENDERER", "tiled")

from scene import sim  # select the renderer before the scene builds

output = Path(__file__).with_name("hurdle_preview.png")
Image.fromarray(sim.render("reference_view")["rgb"]).save(output)
print(output)
