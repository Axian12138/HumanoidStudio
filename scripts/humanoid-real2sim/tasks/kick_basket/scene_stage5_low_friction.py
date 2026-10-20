"""Stage 5 diagnostic: candidate 01 with lower sliding friction."""

import importlib.util
from pathlib import Path

_path = Path(__file__).with_name("scene.py")
_spec = importlib.util.spec_from_file_location("kick_basket_original_scene", _path)
_scene = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_scene)

BASKET_MASS_KG = 1.6
BASKET_FRICTION = 0.15
BASKET_COM_Z_M = 0.12


def build_sim(asset, motion):
    return _scene.build_sim(
        asset,
        motion,
        basket_mass=BASKET_MASS_KG,
        basket_friction=BASKET_FRICTION,
        basket_com_z=BASKET_COM_Z_M,
    )
