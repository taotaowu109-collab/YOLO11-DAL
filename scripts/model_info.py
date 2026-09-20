from __future__ import annotations

import argparse
import csv
from pathlib import Path
from yolo11_dal.model import VARIANTS, variant_summary


def main():
    p = argparse.ArgumentParser(description="Print reconstructed model parameter summaries.")
    p.add_argument("--variant", choices=sorted(VARIANTS), default="full")
    p.add_argument("--all", action="store_true")
    p.add_argument("--csv", default=None)
    args = p.parse_args()

    names = list(VARIANTS) if args.all else [args.variant]
    rows = [variant_summary(name) for name in names]
    columns = ["variant", "parameters", "trainable_parameters", "levels", "strides"]
    print(",".join(columns))
    for row in rows:
        print(",".join(str(row[c]) for c in columns))

    if args.csv:
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
