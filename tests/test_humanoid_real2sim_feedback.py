"""Feedback triage records enough decisions to support future reuse."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

TIER = "fast"
ROOT = Path(__file__).resolve().parents[1]


class FeedbackRoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        spec = importlib.util.spec_from_file_location(
            "humanoid_feedback", ROOT / "scripts/humanoid-real2sim/feedback.py"
        )
        assert spec is not None and spec.loader is not None
        self.feedback = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.feedback)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.feedback.EXAMPLES = Path(self.tmp.name)
        self.feedback.CASES = Path(self.tmp.name) / "feedback"
        self.feedback.CASES.mkdir()
        (self.feedback.EXAMPLES / "index.json").write_text("[]")

    def command(self, *args: str) -> None:
        with patch.object(sys, "argv", ["feedback.py", *args]), redirect_stdout(StringIO()):
            self.feedback.main()

    def test_new_feedback_needs_type_and_three_routing_decisions_before_verification(self) -> None:
        self.command(
            "add",
            "--id",
            "new-grasp",
            "--kind",
            "error",
            "--stage",
            "retarget",
            "--category",
            "interaction",
            "--impact",
            "high",
            "--task",
            "brush",
            "--actions",
            "抓取",
            "--title",
            "手背穿模",
            "--narrative",
            "手背接近把手时穿模",
            "--expected",
            "掌侧接近",
            "--observed",
            "手背先接触",
            "--source",
            "用户截图",
        )
        path = self.feedback.CASES / "new-grasp.json"
        self.assertEqual(json.loads(path.read_text())["kind"], "error")
        with patch("sys.stderr", new=StringIO()), self.assertRaises(SystemExit):
            self.command("update", "new-grasp", "--status", "verified", "--validation", "几何审查通过")
        self.command(
            "update",
            "new-grasp",
            "--cause",
            "腕姿目标错误",
            "--fix",
            "重解掌侧接近",
            "--validation",
            "抓取路径审查通过",
            "--route",
            "skill=none: 动作约束留在样本",
            "--route",
            "tool=update: 修正接近目标",
            "--route",
            "example=update: 加入失败与结果",
            "--status",
            "verified",
        )
        case = json.loads(path.read_text())
        self.assertEqual(case["status"], "verified")
        self.assertEqual(case["routing"]["tool"]["decision"], "update")
        self.assertEqual(case["cause"], "腕姿目标错误")

    def test_legacy_feedback_remains_readable(self) -> None:
        self.assertTrue(self.feedback.ready_to_verify({"fix": "已修", "validation": "已验"}))
        self.assertFalse(self.feedback.ready_to_verify({"kind": "error", "fix": "已修", "validation": "已验"}))


if __name__ == "__main__":
    unittest.main()
