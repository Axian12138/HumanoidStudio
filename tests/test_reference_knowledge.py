"""Prose references do not need binary artifacts; invalid local sources remain errors."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

TIER = "fast"
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("knowledge_audit", ROOT / "scripts/humanoid-real2sim/audit_knowledge.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class TestReferenceKnowledge(unittest.TestCase):
    def test_text_only_reference_is_indexable(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "principle.md").write_text("A reusable decision criterion")
            entry = {
                "id": "text",
                "status": "hypothesis",
                "references": [{"title": "Principle", "target": "principle.md", "why": "Check applicability"}],
            }
            self.assertEqual(audit.audit_references(entry, root), [])

    def test_missing_reference_or_purpose_is_rejected(self):
        entry = {
            "id": "text",
            "status": "hypothesis",
            "references": [{"title": "Missing", "target": "missing.md", "why": "Read source"}],
        }
        with tempfile.TemporaryDirectory() as folder:
            self.assertTrue(audit.audit_references(entry, Path(folder)))
        entry["references"][0]["target"] = "https://example.org/reference"
        del entry["references"][0]["why"]
        self.assertTrue(audit.audit_references(entry, ROOT))

    def test_existing_evidence_contract_remains_accepted(self):
        self.assertEqual(
            audit.audit_references({"id": "legacy", "status": "scoped", "evidence_ids": ["existing"]}, ROOT), []
        )

    def test_new_directory_and_no_old_alias(self):
        self.assertTrue((ROOT / "reference-knowledge/humanoid-real2sim/KNOWLEDGE.md").is_file())
        self.assertFalse((ROOT / "examples").exists())
