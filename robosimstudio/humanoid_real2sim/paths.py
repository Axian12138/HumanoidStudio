"""Stable locations for humanoid real2sim code and example data.

Task programs live under ``scripts/humanoid-real2sim/tasks``. Their recorded
motions, meshes and review artifacts live under ``examples/humanoid-real2sim``.
"""

from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
EXAMPLES = REPOSITORY / "examples" / "humanoid-real2sim"
SCRIPTS = REPOSITORY / "scripts" / "humanoid-real2sim"


def example_dir(script: str | Path) -> Path:
    """Return the data directory matching a task script's relative directory."""
    relative = Path(script).resolve().relative_to(SCRIPTS / "tasks")
    return EXAMPLES / relative.parent
