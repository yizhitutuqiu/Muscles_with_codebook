from __future__ import annotations

from pathlib import Path


def get_musclesinaction_repo_root() -> Path:
    """
    Return repo root for `golf_third_party/musclesinaction/` (the folder that contains
    `MIADatasetOfficial/`, `musclesinaction/`, `custom/`, ...).
    """
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "MIADatasetOfficial").is_dir() and (parent / "musclesinaction").is_dir():
            return parent
    # Fallback: custom/utils/path_utils.py -> custom/utils -> custom -> musclesinaction/
    return p.parents[2]

