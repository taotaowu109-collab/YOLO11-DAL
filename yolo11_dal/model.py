from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import yaml
from ultralytics.nn.tasks import DetectionModel

from .modules import ASFDetect, C3k2LEGM, DAHDetect, infer_detect_input_channels, infer_module_out_channels

ROOT = Path(__file__).resolve().parents[1]
BASELINE_YAML = ROOT / "configs" / "models" / "yolo11n_baseline.yaml"
P2_YAML = ROOT / "configs" / "models" / "yolo11n_p2.yaml"


@dataclass(frozen=True)
class Variant:
    name: str
    use_p2: bool
    use_legm: bool
    use_asf: bool
    use_dah: bool


VARIANTS = {
    "baseline": Variant("baseline", False, False, False, False),
    "p2": Variant("p2", True, False, False, False),
    "dah": Variant("dah", True, False, False, True),
    "legm": Variant("legm", False, True, False, False),
    "asf": Variant("asf", True, False, True, False),
    "dah_legm": Variant("dah_legm", True, True, False, True),
    "dah_asf": Variant("dah_asf", True, False, True, True),
    "legm_asf": Variant("legm_asf", True, True, True, False),
    "full": Variant("full", True, True, True, True),
}


def _load_cfg(path: Path, nc: int) -> dict:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["nc"] = int(nc)
    cfg["yaml_file"] = str(path)
    return cfg


def _copy_graph_meta(dst: torch.nn.Module, src: torch.nn.Module) -> None:
    for key in ("i", "f", "type"):
        if hasattr(src, key):
            setattr(dst, key, getattr(src, key))
    dst.np = sum(p.numel() for p in dst.parameters())


def _replace_legm_blocks(model: DetectionModel, indices: Iterable[int] = (6, 8)) -> None:
    for idx in indices:
        original = model.model[idx]
        channels = infer_module_out_channels(original)
        replacement = C3k2LEGM(original, channels)
        _copy_graph_meta(replacement, original)
        model.model[idx] = replacement


def _replace_head(model: DetectionModel, use_asf: bool, use_dah: bool, asf_channels: int = 64, dah_hidden: int = 64) -> None:
    old = model.model[-1]
    ch = infer_detect_input_channels(old)
    if len(ch) != 4:
        raise ValueError(f"ASF/DAH requires a P2-P5 four-level skeleton, got {len(ch)} levels")

    if use_dah:
        new = DAHDetect(
            nc=old.nc,
            ch=ch,
            reg_max=old.reg_max,
            use_asf=use_asf,
            asf_channels=asf_channels,
            hidden_channels=dah_hidden,
        )
    elif use_asf:
        new = ASFDetect(nc=old.nc, ch=ch, reg_max=old.reg_max, out_channels=asf_channels)
    else:
        return

    _copy_graph_meta(new, old)
    new.stride = old.stride.detach().clone()
    model.model[-1] = new
    model.stride = new.stride
    new.bias_init()


def build_model(
    variant: str = "full",
    nc: int = 7,
    verbose: bool = False,
    asf_channels: int = 64,
    dah_hidden: int = 64,
) -> DetectionModel:
    if variant not in VARIANTS:
        raise KeyError(f"Unknown variant '{variant}'. Choose from: {', '.join(VARIANTS)}")

    spec = VARIANTS[variant]
    cfg_path = P2_YAML if spec.use_p2 else BASELINE_YAML
    model = DetectionModel(cfg=_load_cfg(cfg_path, nc), ch=3, nc=nc, verbose=verbose)
    model.variant = variant

    if spec.use_legm:
        _replace_legm_blocks(model)
    if spec.use_asf or spec.use_dah:
        _replace_head(model, spec.use_asf, spec.use_dah, asf_channels, dah_hidden)

    for module in model.model:
        if hasattr(module, "np"):
            module.np = sum(p.numel() for p in module.parameters())
    return model


def variant_summary(variant: str = "full", nc: int = 7) -> dict:
    model = build_model(variant=variant, nc=nc, verbose=False)
    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "variant": variant,
        "parameters": params,
        "trainable_parameters": trainable,
        "levels": int(len(model.stride)),
        "strides": ",".join(str(int(x)) for x in model.stride.tolist()),
    }
