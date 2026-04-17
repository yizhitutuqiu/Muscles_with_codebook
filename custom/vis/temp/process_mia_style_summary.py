from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple


METHODS: Tuple[str, ...] = ("stage2", "official_cond", "official_nocond", "retrieval")


def _read_rows(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(r) for r in reader]


def _fmt_rmse(v: str | None) -> str:
    if v is None:
        return ""
    s = str(v).strip()
    if not s:
        return ""
    try:
        return f"{float(s):.2f}"
    except Exception:
        return ""


def _is_exercise_name(val_filelist: str) -> bool:
    s = str(val_filelist).strip()
    if not s.startswith("val"):
        return False
    if "Subject" in s:
        return False
    if "_" in s:
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=str,
        default="/data/litengmo/HSMR/mia_custom/custom/output/mia_style_eval/20260402_162050/summary.csv",
    )
    args = parser.parse_args()

    in_path = Path(args.csv).expanduser().resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"Input csv not found: {in_path}")

    rows = _read_rows(in_path)

    table: Dict[str, Dict[str, str]] = {}
    for r in rows:
        proto = str(r.get("protocol", "")).strip()
        val_name = str(r.get("val_filelist", "")).strip()
        method = str(r.get("method", "")).strip()
        rmse = r.get("official_global_rmse", None)

        if proto == "id_exercises" or _is_exercise_name(val_name):
            if method not in METHODS:
                continue
            if val_name not in table:
                table[val_name] = {}
            table[val_name][method] = _fmt_rmse(rmse)

    out_path = in_path.parent / "summary_processed.csv"
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["exercise", *METHODS])
        for ex in sorted(table.keys()):
            row = [ex]
            for m in METHODS:
                row.append(table[ex].get(m, ""))
            writer.writerow(row)

    print(str(out_path))


if __name__ == "__main__":
    main()
