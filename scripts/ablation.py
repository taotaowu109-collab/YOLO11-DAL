from __future__ import annotations

import argparse
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description="Run the 2x2x2 ablation plus optional P2-only control.")
    p.add_argument("--data", default="configs/data/powerline.yaml")
    p.add_argument("--device", default=None)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--include-p2-control", action="store_true")
    args = p.parse_args()

    order = ["baseline", "dah", "legm", "asf", "dah_legm", "dah_asf", "legm_asf", "full"]
    if args.include_p2_control:
        order.insert(1, "p2")

    for variant in order:
        cmd = [
            sys.executable, "scripts/train.py",
            "--variant", variant,
            "--data", args.data,
            "--epochs", str(args.epochs),
            "--batch", str(args.batch),
            "--name", variant,
        ]
        if args.device is not None:
            cmd += ["--device", args.device]
        print("\n>>>", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
