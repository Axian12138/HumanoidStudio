"""Humanoid real2sim tools and examples have one canonical home."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

TIER = "fast"
ROOT = Path(__file__).resolve().parents[1]


class HumanoidReal2SimLayoutTests(unittest.TestCase):
    def test_only_canonical_tool_package_remains(self) -> None:
        from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields
        from robosimstudio.humanoid_real2sim.track.runtime import RepairRuntime
        from robosimstudio.humanoid_real2sim.track.timeline import ReferenceMotion
        from robosimstudio.humanoid_real2sim.track.tracker import BFMTracker

        self.assertTrue(all((MotionFields, RepairRuntime, ReferenceMotion, BFMTracker)))
        self.assertFalse((ROOT / "robosimstudio/humanoid_repair").exists())
        self.assertFalse((ROOT / "robosimstudio/real2sim").exists())

    def test_curated_examples_have_narratives(self) -> None:
        directory = ROOT / "reference-knowledge/humanoid-real2sim"
        index = json.loads((directory / "index.json").read_text())
        self.assertTrue(index)
        for item in index:
            with self.subTest(example=item["id"]):
                narrative = (directory / item["file"]).read_text()
                self.assertTrue(item["actions"])
                self.assertGreater(len(narrative), 300)
                self.assertTrue(narrative.startswith("# "))

    def test_only_canonical_example_path_remains(self) -> None:
        self.assertTrue((ROOT / "reference-knowledge/humanoid-real2sim/KNOWLEDGE.md").is_file())
        self.assertFalse((ROOT / "reference-knowledge/humanoid-real2sim/brush_sweep_1019").exists())
        self.assertFalse((ROOT / "reference-knowledge/humanoid_motion_repair").exists())
        self.assertFalse((ROOT / "reference-knowledge/real2sim").exists())

    def test_code_and_example_data_are_separate(self) -> None:
        from robosimstudio.humanoid_real2sim.paths import example_dir

        samples = ROOT / "reference-knowledge/humanoid-real2sim"
        self.assertFalse([p for p in samples.rglob("*") if p.suffix in {".py", ".sh", ".html"}])
        script = ROOT / "scripts/humanoid-real2sim/tasks/brush_sweep_1019/scene.py"
        self.assertTrue(script.is_file())
        data = ROOT / "outputs/humanoid-real2sim/2026-10-20"
        self.assertEqual(example_dir(script), data / "brush_sweep_1019/curated_snapshot")
        nested = ROOT / "scripts/humanoid-real2sim/tasks/sequence_1019/tasks/06_round_basket/scene.py"
        self.assertEqual(example_dir(nested), data / "sequence_1019/curated_snapshot/tasks/06_round_basket")


if __name__ == "__main__":
    unittest.main()
