from __future__ import annotations

import argparse
from ultralytics.models.yolo.detect import DetectionTrainer
from yolo11_dal.model import BASELINE_YAML, P2_YAML, VARIANTS, build_model


def parse_args():
    p = argparse.ArgumentParser(description="Train a YOLO11-DAL ablation variant.")
    p.add_argument("--variant", default="full", choices=sorted(VARIANTS))
    p.add_argument("--data", default="configs/data/powerline.yaml")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--lr0", type=float, default=0.01)
    p.add_argument("--momentum", type=float, default=0.937)
    p.add_argument("--weight-decay", type=float, default=0.0005)
    p.add_argument("--optimizer", default="SGD")
    p.add_argument("--device", default=None)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--project", default="runs/yolo11_dal")
    p.add_argument("--name", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--deterministic", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    skeleton = P2_YAML if VARIANTS[args.variant].use_p2 else BASELINE_YAML
    overrides = {
        "model": str(skeleton),
        "data": args.data,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "lr0": args.lr0,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "optimizer": args.optimizer,
        "workers": args.workers,
        "project": args.project,
        "name": args.name or args.variant,
        "seed": args.seed,
        "deterministic": args.deterministic,
        "pretrained": False,
        "task": "detect",
    }
    if args.device is not None:
        overrides["device"] = args.device

    trainer = DetectionTrainer(overrides=overrides)
    trainer.model = build_model(args.variant, nc=7, verbose=True)
    trainer.train()


if __name__ == "__main__":
    main()
