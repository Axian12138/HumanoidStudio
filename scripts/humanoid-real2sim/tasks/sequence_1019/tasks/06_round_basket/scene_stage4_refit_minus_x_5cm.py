"""Stage 4 scene using the basket path refit with fixed humanoid motion."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[6]
OBJECT_TRACK = (
    ROOT
    / "outputs/humanoid-real2sim/2026-10-20/sequence_1019/runs/stage4_round_basket_refit_minus_x_5cm_01/retarget/object_estimate.npz"
)


def build_sim(asset, motion):
    source = Path(__file__).with_name("scene_stage4.py")
    spec = importlib.util.spec_from_file_location("round_basket_stage4_base", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Stage 4 basket scene: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_sim(
        asset,
        motion,
        basket_mass_kg=1.6,
        basket_com_z_m=0.08,
        basket_track_path=OBJECT_TRACK,
    )
