"""Reuse the existing MCAP rigid/calibration extractor for clip 10's own source window."""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--run", type=Path, required=True)
a = p.parse_args()
root = Path(__file__).resolve().parents[6]
spec = importlib.util.spec_from_file_location(
    "rigid_extractor", Path(__file__).parents[1] / "09_lined_rect_basket/extract_wrist_reference.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m.TASK = a.run / "retarget/body_prior/10_flower_bouquet"
m.SNAPSHOT = root / "outputs/humanoid-real2sim/2026-10-20/sequence_1019/curated_snapshot"
sys.argv = [__file__, "--out", str(a.run / "inputs/10_flower_bouquet/rigid")]
m.main()
# Upstream task-09 extractor hardcodes metadata [98,109]; correct only this output.

out = a.run / "inputs/10_flower_bouquet/rigid"
path = out / "wrist_reference.npz"
with np.load(path) as z:
    values = {k: z[k] for k in z.files}
values["source_range_s"] = values["recording_time_s"][[0, -1]]
np.savez_compressed(path, **values)
f = out / "wrist_reference.json"
report = json.loads(f.read_text())
report["source_range_s"] = values["source_range_s"].tolist()
report["metadata_correction"] = "task-09 hardcoded range replaced by actual recording_time_s endpoints"
f.write_text(json.dumps(report, indent=2))
