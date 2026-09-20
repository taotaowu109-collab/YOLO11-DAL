from __future__ import annotations

import argparse
from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser(description="Run YOLO11-DAL inference.")
    p.add_argument("weights")
    p.add_argument("source")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--device", default=None)
    p.add_argument("--save", action="store_true")
    args = p.parse_args()

    model = YOLO(args.weights, task="detect")
    kwargs = {"source": args.source, "imgsz": args.imgsz, "conf": args.conf, "save": args.save}
    if args.device is not None:
        kwargs["device"] = args.device
    model.predict(**kwargs)


if __name__ == "__main__":
    main()
