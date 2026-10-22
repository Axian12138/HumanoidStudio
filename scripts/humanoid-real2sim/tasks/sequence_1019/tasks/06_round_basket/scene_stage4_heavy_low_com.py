"""Heavier, lower-center-of-mass round-basket Stage 4 physics candidate."""

import importlib.util
from pathlib import Path


BASKET_MASS_KG = 1.6
BASKET_COM_Z_M = 0.08


def build_sim(asset, motion):
    source = Path(__file__).with_name("scene_stage4.py")
    spec = importlib.util.spec_from_file_location("round_basket_stage4_base", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Stage 4 basket scene: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_sim(
        asset, motion,
        basket_mass_kg=BASKET_MASS_KG,
        basket_com_z_m=BASKET_COM_Z_M,
    )
