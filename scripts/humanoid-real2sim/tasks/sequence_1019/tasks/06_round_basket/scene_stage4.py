"""Physical Stage 4 baseline for the 59-76 s round-basket retarget candidate."""

from pathlib import Path

import newton
import numpy as np

import robosimstudio as rss
from robosimstudio.core.tuning import notify
from robosimstudio.humanoid_real2sim.paths import example_dir
from robosimstudio.scene.entity import label_index

TASK = example_dir(__file__)
SNAPSHOT = TASK.parents[1]
KICK = SNAPSHOT.parents[1] / "kick_basket/curated_snapshot/assets"
HULLS = tuple(sorted((KICK / "collision_hulls").glob("Hull_*.obj")))
OBJECT_TRACK = TASK / "hoi_v5_kneel_59_76/object_estimate.npz"
TABLE = SNAPSHOT / "assets/morbylanga_table_1012.obj"


def build_sim(
    asset: Path,
    motion,
    *,
    basket_mass_kg: float = 0.8,
    basket_com_z_m: float | None = None,
    basket_track_path: Path = OBJECT_TRACK,
):
    if len(HULLS) != 65:
        raise ValueError(f"expected 65 round-basket collision hulls, found {len(HULLS)}")
    with np.load(basket_track_path, allow_pickle=False) as track:
        initial = tuple(float(v) for v in track["position_w"][0])
        if int(track["fps"]) != int(motion.fps) or len(track["position_w"]) != motion.frames:
            raise ValueError("basket estimate does not match the Stage 4 reference")
    g1 = rss.EmbodimentSpec(
        name="g1_round_basket_stage4",
        asset=rss.AssetSpec(path=str(asset)),
        preserve_free_bodies=True,
        enable_self_collisions=False,
        arms=(
            rss.ArmSpec(
                name="whole_body",
                joint_names=motion.joint_names,
                ee_body_name="right_wrist_yaw_link",
                home_q=tuple(float(v) for v in motion.joint_pos[0]),
            ),
        ),
    )
    scene = rss.Scene(cell="void", name="round_basket_1019_stage4", physics=rss.MUJOCO_500HZ)
    scene.add(rss.robot(g1))
    scene.add(
        rss.StaticSpec(
            name="morbylanga_table_context",
            asset=rss.AssetSpec(path=str(TABLE)),
            world_xform=True,
            collide=False,
        )
    )
    scene.add(
        rss.body(
            name="round_laundry_basket",
            asset=rss.AssetSpec(
                path=str(TASK / "round_laundry_basket_visual.obj"),
                collision_paths=tuple(str(path) for path in HULLS),
            ),
            at=initial,
            mass=basket_mass_kg,
            friction=0.8,
            color=(0.60, 0.43, 0.29),
        )
    )
    scene.add(
        rss.CameraSpec(
            name="review",
            width=960,
            height=540,
            eye=(1.4, -3.4, 1.5),
            look_at=(0.05, -1.9, 0.45),
        )
    )
    sim = scene.build(control_mode="joint_position")
    if basket_com_z_m is not None:
        basket = label_index(list(sim.model.body_label), "round_laundry_basket")
        center = np.asarray(sim.model.body_com.numpy()).copy()
        center[basket, 2] = basket_com_z_m
        sim.model.body_com.assign(center)
        notify(sim.pipeline, int(newton.ModelFlags.BODY_INERTIAL_PROPERTIES))
    return sim
