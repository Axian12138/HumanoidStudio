"""Install verified motion-only changes with source hashes and separate backups."""

import hashlib
import json
import os
import shutil
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    root = Path("/home/galbot/workspace/ScaleTrack/data/1019test")
    out = Path("/home/galbot/workspace/robosimstudio/outputs/humanoid-real2sim/2026-10-25/table_hand_clearance/runs")
    validation = json.loads((out / "validation.json").read_text())
    assert len(validation["cases"]) == 4
    cases = [
        ("thumb", "push-armchair-thumb-kinematic-v1", "motion.npz"),
        ("index-middle", "push-armchair-index-middle-kinematic-v1", "motion.npz"),
        ("paired_thumb", "brace-dining-table-wipes-kinematic-v1", "motion.npz"),
        ("paired_thumb", "brace-dining-table-wipes-kinematic-v1", "motion_after_thumb.npz"),
        ("paired_index-middle", "brace-dining-table-wipes-kinematic-v1", "motion_after_index_middle.npz"),
    ]
    for _, name, file in cases:
        dest = root / name / file
        if file == "motion.npz":
            assert digest(dest) == digest(out / "original_backup" / name / file), "source changed since backup"
        else:
            assert not dest.exists(), f"already exists: {dest}"
    installed = []
    for run, name, file in cases:
        src = out / run / "motion.npz"
        dest = root / name / file
        tmp = dest.with_suffix(".npz.tmp")
        shutil.copy2(src, tmp)
        expected = digest(src)
        assert digest(tmp) == expected
        os.replace(tmp, dest)
        installed.append({"path": str(dest), "sha256": expected})
    (out / "installed.json").write_text(json.dumps(installed, indent=2))
    print(json.dumps(installed, indent=2))


if __name__ == "__main__":
    main()
