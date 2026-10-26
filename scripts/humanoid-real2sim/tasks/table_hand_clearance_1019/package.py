"""Replace only the seven motion arrays, retaining compressed extra members."""

import argparse
import json
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--core", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(a.source, a.out)
    with tempfile.TemporaryDirectory(prefix="rss_motion_members_") as tmp:
        z = np.load(a.core)
        for k in z.files:
            np.save(Path(tmp) / (k + ".npy"), z[k], allow_pickle=False)
        subprocess.run(["zip", "-q", str(a.out.resolve()), *[k + ".npy" for k in z.files]], cwd=tmp, check=True)
    with zipfile.ZipFile(a.source) as old, zipfile.ZipFile(a.out) as new:
        assert set(old.namelist()) == set(new.namelist())
        changed = [name for name in old.namelist() if old.getinfo(name).CRC != new.getinfo(name).CRC]
        assert set(changed) <= {k + ".npy" for k in z.files}
    (a.out.parent / "members.json").write_text(
        json.dumps({"changed_members": changed, "all_other_members_identical": True}, indent=2)
    )
    print(a.out, flush=True)


if __name__ == "__main__":
    main()
