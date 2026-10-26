"""Check prose reference knowledge, optional provenance and time-first output storage."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
EXAMPLES = REPOSITORY / "reference-knowledge/humanoid-real2sim"
CASES = EXAMPLES
OUTPUTS = REPOSITORY / "outputs/humanoid-real2sim"
LINK = re.compile(r"(?<!!)\[[^]]+\]\(([^()]+)\)")
DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def audit_references(item: dict, repository: Path) -> list[str]:
    """Prose can stand on useful references; artifacts and code are optional."""
    errors = []
    if not item.get("status"):
        errors.append(f"knowledge missing status: {item['id']}")
    if not item.get("references") and not item.get("evidence_ids"):
        errors.append(f"knowledge missing references or legacy evidence: {item['id']}")
    for ref in item.get("references", []):
        if not all(ref.get(key) for key in ("title", "target", "why")):
            errors.append(f"reference lacks title, target or reading purpose: {item['id']}")
            continue
        target = ref["target"]
        if not target.startswith(("https://", "http://")) and not (repository / target).exists():
            errors.append(f"missing reference: {item['id']} -> {target}")
    return errors


def audit(*, require_archive: bool) -> list[str]:
    errors = []
    index = json.loads((CASES / "index.json").read_text())
    evidence = json.loads((EXAMPLES / "evidence_manifest.json").read_text())
    artifacts = {item["id"]: item for item in evidence["artifacts"]}
    if len(artifacts) != len(evidence["artifacts"]):
        errors.append("duplicate evidence id")
    result_manifest = json.loads((EXAMPLES / "results_manifest.json").read_text())
    results = {item["id"]: item for item in result_manifest["results"]}
    if len(results) != len(result_manifest["results"]):
        errors.append("duplicate result id")
    ids = [item["id"] for item in index]
    if len(ids) != len(set(ids)):
        errors.append("duplicate case id")
    for item in index:
        path = CASES / item["file"]
        if not path.is_file():
            errors.append(f"missing indexed case: {path}")
        if Path(item["file"]).parts[0] != item.get("category"):
            errors.append(f"category/file mismatch: {item['id']}")
        errors.extend(audit_references(item, REPOSITORY))
        for code_path in item.get("code_paths", []):
            if not (REPOSITORY / code_path).is_file():
                errors.append(f"missing implementation path: {item['id']} -> {code_path}")
        for evidence_id in item.get("evidence_ids", []):
            artifact = artifacts.get(evidence_id)
            if artifact is None:
                errors.append(f"unknown evidence id: {item['id']} -> {evidence_id}")

        for result_id in item.get("result_ids", []):
            result = results.get(result_id)
            if result is None:
                errors.append(f"unknown result id: {item['id']} -> {result_id}")
            elif item["id"] not in result.get("case_ids", []):
                errors.append(f"result/case mismatch: {item['id']} -> {result_id}")

    referenced = {eid for item in index for eid in item.get("evidence_ids", [])}
    for artifact in evidence["artifacts"]:
        path = EXAMPLES / artifact["file"]
        if artifact["id"] not in referenced:
            errors.append(f"unreferenced evidence: {artifact['id']}")
        if "artifacts" not in Path(artifact["file"]).parts:
            errors.append(f"evidence outside artifacts directory: {artifact['id']}")
        if path.suffix.lower() not in {".npz", ".png", ".jpg", ".jpeg", ".json"}:
            errors.append(f"unsupported evidence type: {artifact['id']}")
        if not artifact.get("source") or not artifact.get("status") or not artifact.get("meaning"):
            errors.append(f"evidence lacks provenance or scope: {artifact['id']}")
        if not path.is_file():
            errors.append(f"missing evidence file: {artifact['file']}")
        else:
            if path.stat().st_size != artifact["bytes"]:
                errors.append(f"evidence size changed: {artifact['id']}")
            if hashlib.sha256(path.read_bytes()).hexdigest() != artifact["sha256"]:
                errors.append(f"evidence hash changed: {artifact['id']}")
    referenced_results = {rid for item in index for rid in item.get("result_ids", [])}
    for result in result_manifest["results"]:
        if result["id"] not in referenced_results:
            errors.append(f"unreferenced result: {result['id']}")
        if "results" not in Path(result["file"]).parts or not result.get("accepted_scope"):
            errors.append(f"result lacks location or accepted scope: {result['id']}")
        if not result.get("claim") or not result.get("source") or not result.get("review_evidence_ids"):
            errors.append(f"result lacks claim, provenance or review: {result['id']}")
        if result.get("frames", 0) <= 0 or result.get("fps", 0) <= 0:
            errors.append(f"result lacks frame count or fps: {result['id']}")
        for evidence_id in result.get("review_evidence_ids", []):
            if evidence_id not in artifacts:
                errors.append(f"result references unknown review: {result['id']} -> {evidence_id}")
        for entry in [result, *result.get("companions", [])]:
            path = EXAMPLES / entry["file"]
            if not path.is_file():
                errors.append(f"missing result file: {entry['file']}")
            else:
                if path.stat().st_size != entry["bytes"]:
                    errors.append(f"result size changed: {entry['file']}")
                if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
                    errors.append(f"result hash changed: {entry['file']}")
    manifest = json.loads((EXAMPLES / "curation_manifest.json").read_text())
    documents = list(EXAMPLES.rglob("*.md"))
    if require_archive:
        for info in manifest["tasks"].values():
            documents.extend((REPOSITORY / info["selected_data"]).rglob("*.md"))
    for path in documents:
        for target in LINK.findall(path.read_text()):
            if target.startswith(("http:", "https:", "#", "/", "mailto:")):
                continue
            resolved = (path.parent / target.split("#", 1)[0]).resolve()
            if not resolved.exists() and (require_archive or not resolved.is_relative_to(OUTPUTS)):
                errors.append(f"broken link: {path.relative_to(REPOSITORY)} -> {target}")
    tasks = {"brush_pick_1012", "brush_sweep_1019", "push_bin_1019", "kick_basket"}
    segments = json.loads((EXAMPLES / "task-analysis/artifacts/1019_segments.json").read_text())
    tasks.update("sequence_1019/" + item["id"] for item in segments["tasks"])
    represented = " ".join(item["task"] for item in index)
    for task in sorted(tasks):
        if task not in represented:
            errors.append(f"task has no knowledge entry: {task}")
    for path in OUTPUTS.iterdir() if OUTPUTS.exists() else []:
        if path.is_dir() and DATE.fullmatch(path.name) is None:
            errors.append(f"output root must be a date: {path}")
    for task in manifest["tasks"]:
        if (EXAMPLES / task).exists():
            errors.append(f"task data must be in outputs: {task}")
    for path in EXAMPLES.rglob("rollouts"):
        if path.is_dir():
            errors.append(f"full rollout must be in outputs: {path}")
    for task, info in manifest["tasks"].items():
        if require_archive:
            selected = REPOSITORY / info["selected_data"]
            for name in info["preserved_paths"]:
                if not (selected / name).is_file():
                    errors.append(f"missing curated file: {task}/{name}")
            if not (REPOSITORY / info["archive"]).is_dir():
                errors.append(f"missing local output archive: {info['archive']}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-archive", action="store_true", help="also require the local ignored historical archive"
    )
    args = parser.parse_args()
    errors = audit(require_archive=args.require_archive)
    if errors:
        for error in errors:
            print("FAIL", error)
        raise SystemExit(1)
    print(
        "PASS reference knowledge, task coverage, optional result/evidence hashes, links, output layout"
        + (", local archive" if args.require_archive else "")
    )


if __name__ == "__main__":
    main()
