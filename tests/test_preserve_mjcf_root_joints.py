"""Preserving free bodies must not convert authored mechanisms to free joints."""

import tempfile
import unittest
from pathlib import Path

import newton

from robosimstudio.robots.entity import EntityEmbodiment
from robosimstudio.specs import AssetSpec, EmbodimentSpec

TIER = "fast"


class TestPreserveRootJoints(unittest.TestCase):
    def test_mixed_free_slider_and_fixed_roots_keep_their_types(self):
        source = """<mujoco><worldbody>
          <body name="robot"><freejoint name="floating_base"/><geom size=".1"/>
            <body name="arm" pos="0 0 .2"><joint name="elbow"/><geom size=".05"/></body>
          </body>
          <body name="fruit" pos="1 0 0"><freejoint name="fruit_free_joint"/><geom size=".05"/></body>
          <body name="drawer" pos="2 0 0"><joint name="drawer_slide" type="slide" axis="0 1 0"/>
            <geom type="box" size=".2 .2 .1"/></body>
          <body name="cabinet" pos="3 0 0"><geom type="box" size=".3 .3 .3"/></body>
        </worldbody></mujoco>"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mixed.xml"
            path.write_text(source)
            for collapse in (False, True):
                with self.subTest(collapse=collapse):
                    builder = newton.ModelBuilder()
                    entity = EntityEmbodiment(
                        EmbodimentSpec(
                            name="mixed",
                            asset=AssetSpec(path=str(path)),
                            arms=(),
                            preserve_free_bodies=True,
                            collapse_fixed_joints=collapse,
                        )
                    )
                    entity.build(builder, None)
                    labels = [label.split("/")[-1] for label in builder.joint_label]
                    self.assertEqual(labels.count("floating_base"), 1)
                    for name, kind in (
                        ("floating_base", newton.JointType.FREE),
                        ("fruit_free_joint", newton.JointType.FREE),
                        ("drawer_slide", newton.JointType.PRISMATIC),
                    ):
                        self.assertEqual(builder.joint_type[labels.index(name)], kind)
                    self.assertEqual(builder.joint_type.count(newton.JointType.FREE), 2)
