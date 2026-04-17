"""
Build filelist of AMASS npz paths for Stage1 pre-training (cmu + hdm05).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass(frozen=True)
class AmassScanResult:
    filelist_path: Path
    num_files: int


def collect_amass_npz(
    root: Path,
    subdirs: List[str],
    *,
    suffix: str = "_stageii.npz",
    max_files: Optional[int] = None,
) -> List[Path]:
    """
    Collect all npz under root/subdir/... that end with suffix (e.g. _stageii.npz).
    Skips neutral_stagei.npz and similar.
    """
    out: List[Path] = []
    for sub in subdirs:
        d = root / sub
        if not d.exists():
            continue
        for p in sorted(d.rglob("*.npz")):
            if not p.name.endswith(suffix):
                continue
            if "neutral" in p.name.lower() or "stagei.npz" in p.name:
                continue
            out.append(p.resolve())
            if max_files is not None and len(out) >= max_files:
                return out
    return out


def build_amass_filelist(
    *,
    amass_root: Path,
    out_txt: Path,
    subdirs: Optional[List[str]] = None,
    max_files: Optional[int] = None,
    suffix: str = "_stageii.npz",
) -> AmassScanResult:
    """
    Write a filelist of AMASS npz paths (one per line) for cmu and hdm05.

    :param amass_root: Root directory containing e.g. cmu/, hdm05/.
    :param out_txt: Output .txt path.
    :param subdirs: Subdirs to scan (default: ["cmu", "hdm05"]).
    :param max_files: Cap number of files (None = no cap).
    :param suffix: Only include npz with this suffix (default: _stageii.npz).
    """
    if subdirs is None:
        subdirs = ["cmu", "hdm05"]
    out_txt = Path(out_txt)
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    paths = collect_amass_npz(Path(amass_root), subdirs, suffix=suffix, max_files=max_files)
    lines = [str(p) for p in paths]
    out_txt.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return AmassScanResult(filelist_path=out_txt, num_files=len(lines))
