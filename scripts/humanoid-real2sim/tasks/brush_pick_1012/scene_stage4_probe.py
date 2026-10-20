"""Stage4 brush placement after free settling, for wrist-pose comparison."""

import importlib.util
from pathlib import Path

source = Path(__file__).with_name("scene.py")
spec = importlib.util.spec_from_file_location("brush_pick_scene_base", source)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def build_sim(asset, motion):
    return base.build_sim(
        asset, motion,
        brush_position=(0.86772203445, -0.61350262165, 0.44350054860),
        brush_quat=(0.0020032993, 0.16367821395, -0.0057759280, 0.9864948392),
    )
