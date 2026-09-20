from __future__ import annotations

import argparse
import torch
from yolo11_dal.model import VARIANTS, build_model, variant_summary


def main():
    p = argparse.ArgumentParser(description="Construct a variant and run a no-grad forward pass.")
    p.add_argument("--variant", default="full", choices=sorted(VARIANTS))
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    model = build_model(args.variant, nc=7, verbose=False).to(args.device).eval()
    x = torch.zeros(1, 3, args.imgsz, args.imgsz, device=args.device)
    with torch.no_grad():
        y = model(x)
    print(variant_summary(args.variant))
    print("prediction shape:", tuple(y[0].shape) if isinstance(y, tuple) else type(y).__name__)


if __name__ == "__main__":
    main()
