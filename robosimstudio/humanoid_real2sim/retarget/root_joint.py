"""Find the reference root by its body, independent of importer joint naming."""

import numpy as np

from robosimstudio.scene.entity import label_index


def root_joint_index(model, root_body: str) -> int:
    """Return the unique world-attached seven-coordinate joint of root_body."""
    body = label_index(list(model.body_label), root_body)
    children = model.joint_child.numpy()
    parents = model.joint_parent.numpy()
    starts = model.joint_q_start.numpy()
    hits = np.flatnonzero((children == body) & (parents < 0) & (np.diff(starts) == 7))
    if len(hits) != 1:
        raise ValueError(f"{root_body!r} must have one world-attached free joint, got {len(hits)}")
    return int(hits[0])
