"""Locations for humanoid knowledge and date-first experiment runs."""

import re
from datetime import datetime
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
EXAMPLES = REPOSITORY / "reference-knowledge" / "humanoid-real2sim"
OUTPUTS = REPOSITORY / "outputs" / "humanoid-real2sim"
SCRIPTS = REPOSITORY / "scripts" / "humanoid-real2sim"
LEGACY_DATA_DATE = "2026-10-20"


def example_dir(script: str | Path) -> Path:
    """Return the moved data snapshot for a historical task script.

    Historical scripts may still read/write this snapshot. New experiment
    results should use :func:`run_dir` instead.
    """
    relative = Path(script).resolve().relative_to(SCRIPTS / "tasks")
    task, *rest = relative.parent.parts
    return OUTPUTS / LEGACY_DATA_DATE / task / "curated_snapshot" / Path(*rest)


def run_dir(task: str, *, when: datetime | None = None, create: bool = True) -> Path:
    """Return ``outputs/humanoid-real2sim/<date>/<task>/runs/<run-id>``.

    A nested task id such as ``sequence_1019/06_round_basket`` becomes one
    second-level directory named ``sequence_1019__06_round_basket``. The time
    first-level directory is the local calendar date. A run id within the
    task directory keeps launches distinct without increasing date precision.
    """
    parts = task.split("/")
    if not parts or any(re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", part) is None for part in parts):
        raise ValueError(f"invalid task id: {task!r}")
    instant = when or datetime.now().astimezone()
    if instant.tzinfo is None or instant.utcoffset() is None:
        raise ValueError("run timestamp must include a timezone")
    instant = instant.astimezone()
    date = instant.strftime("%Y-%m-%d")
    run_id = instant.strftime("%H-%M-%S.%f%z")
    path = OUTPUTS / date / "__".join(parts) / "runs" / run_id
    if create:
        path.mkdir(parents=True, exist_ok=False)
    return path
