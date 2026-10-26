"""USD boolean semantics must survive YAML overrides and importer conversion."""

import tempfile
import unittest
from pathlib import Path

import yaml

TIER = "fast"
try:
    from pxr import Usd, UsdGeom, UsdPhysics
except ImportError:
    Usd = None


@unittest.skipIf(Usd is None, "requires optional USD importer dependencies")
class PackageSceneTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        room = Usd.Stage.CreateNew(str(self.root / "room.usda"))
        room.SetDefaultPrim(UsdGeom.Xform.Define(room, "/Room").GetPrim())
        room.GetRootLayer().Save()
        obj = Usd.Stage.CreateNew(str(self.root / "object.usda"))
        body = UsdGeom.Xform.Define(obj, "/Asset").GetPrim()
        obj.SetDefaultPrim(body)
        UsdPhysics.RigidBodyAPI.Apply(body).CreateKinematicEnabledAttr().Set(False)
        UsdPhysics.MassAPI.Apply(body).CreateMassAttr().Set(1)
        sphere = UsdGeom.Sphere.Define(obj, "/Asset/sphere")
        sphere.CreateRadiusAttr().Set(0.1)
        UsdPhysics.CollisionAPI.Apply(sphere.GetPrim()).CreateCollisionEnabledAttr().Set(True)
        obj.GetRootLayer().Save()
        self.original = (self.root / "object.usda").read_bytes()

    def compose(self, mode, collision):
        from robosimstudio.humanoid_real2sim.track.package_scene import compose_stage

        obj = {
            "id": "prop",
            "usd_path": "object.usda",
            "initial_position_m": [0, 0, 1],
            "initial_orientation_wxyz": [1, 0, 0, 0],
            "scale": [1, 1, 1],
        }
        if mode is not None:
            obj["physics_mode"] = mode
        if collision is not None:
            obj["collision_enabled"] = collision
        (self.root / "motions.yaml").write_text(
            yaml.safe_dump({"take": {"scene_setup": {"scene": {"usd_path": "room.usda"}, "objects": [obj]}}})
        )
        return compose_stage(self.root, self.root, self.root / f"out-{mode}-{collision}")

    def test_all_boolean_combinations_reach_imported_model(self):
        import newton

        from robosimstudio.humanoid_real2sim.track.package_scene import audit_model

        for mode in ("dynamic", "kinematic"):
            for collision in (False, True):
                with self.subTest(mode=mode, collision=collision):
                    path, contract = self.compose(mode, collision)
                    builder = newton.ModelBuilder()
                    builder.add_usd(str(path))
                    report = audit_model(builder, contract)
                    self.assertEqual(report["errors"], [])
                    self.assertEqual(report["bodies"][0]["kinematic"], mode == "kinematic")
                    self.assertEqual((self.root / "object.usda").read_bytes(), self.original)

    def test_unspecified_flags_inherit_usd(self):
        _, contract = self.compose(None, None)
        self.assertEqual(contract["bodies"], {"/World/prop": False})
        self.assertEqual(contract["colliders"], {"/World/prop/sphere": True})

    def test_lost_flags_are_rejected(self):
        import newton

        from robosimstudio.humanoid_real2sim.track.package_scene import audit_model

        path, contract = self.compose("kinematic", False)
        builder = newton.ModelBuilder()
        builder.add_usd(str(path))
        builder.body_flags[0] = int(newton.BodyFlags.DYNAMIC)
        builder.shape_flags[0] |= int(newton.ShapeFlags.COLLIDE_SHAPES)
        report = audit_model(builder, contract)
        self.assertEqual(report["status"], "fail")
        self.assertTrue(any("kinematic" in e for e in report["errors"]))
        self.assertTrue(any("collision" in e for e in report["errors"]))

    def test_string_collision_flag_rejected(self):
        with self.assertRaisesRegex(ValueError, "YAML boolean"):
            self.compose("kinematic", "false")
