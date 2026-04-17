from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


@dataclass(frozen=True)
class MIAScanResult:
    filelist_path: Path
    num_samples: int


def iter_mia_sample_dirs(split_root: Path) -> Iterable[Path]:
    """
    Yield sample directories under MIADatasetOfficial/{split}/... in a deterministic order:
      split_root/Subject*/Action*/<sample_id>/
    """
    if not split_root.exists():
        return
    subjects = sorted([p for p in split_root.iterdir() if p.is_dir() and p.name.startswith("Subject")])
    for subj in subjects:
        actions = sorted([p for p in subj.iterdir() if p.is_dir()])
        for act in actions:
            # sample dirs are numeric names, but we don't assume strict numeric.
            for sd in sorted([p for p in act.iterdir() if p.is_dir()], key=lambda x: x.name):
                yield sd


def build_mia_train_filelist(
    *,
    mia_repo_root: Path,
    split: str = "train",
    out_txt: Path,
    max_samples: Optional[int] = None,
    require_files: Optional[List[str]] = None,
) -> MIAScanResult:
    """
    Create a filelist txt for musclesinaction's official dataloader.

    Important:
    - The official dataloader parses `person = filepath.split('/')[2]` and expects that to be `SubjectX`.
    - Therefore, each line must look like: `MIADatasetOfficial/train/Subject0/Running/5119`
      (so split('/') => ['MIADatasetOfficial','train','Subject0',...]).
    - The file paths are intended to be resolved with cwd = `golf_third_party/musclesinaction/`.
    """
    if require_files is None:
        require_files = ["emgvalues.npy", "joints3d.npy", "pose.npy"]

    out_txt = Path(out_txt)
    out_txt.parent.mkdir(parents=True, exist_ok=True)

    split_root = Path(mia_repo_root) / "MIADatasetOfficial" / split
    if not split_root.exists():
        raise FileNotFoundError(f"Split dir not found: {split_root}")

    lines: List[str] = []
    for sd in iter_mia_sample_dirs(split_root):
        ok = True
        for fn in require_files:
            if not (sd / fn).exists():
                ok = False
                break
        if not ok:
            continue

        rel = sd.relative_to(mia_repo_root)  # MIADatasetOfficial/train/SubjectX/...
        lines.append(str(rel))
        if max_samples is not None and len(lines) >= int(max_samples):
            break

    out_txt.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return MIAScanResult(filelist_path=out_txt, num_samples=len(lines))

