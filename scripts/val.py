from __future__ import annotations

import argparse
from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser(description="Validate a trained YOLO11-DAL checkpoint.")
    p.add_argument("weights")
    p.add_argument("--data", default="configs/data/powerline.yaml")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    model = YOLO(args.weights, task="detect")
    kwargs = {"data": args.data, "imgsz": args.imgsz, "batch": args.batch}
    if args.device is not None:
        kwargs["device"] = args.device
    model.val(**kwargs)


if __name__ == "__main__":
    main()
