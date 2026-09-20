from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d

from ultralytics.nn.modules import Detect


class Scale(nn.Module):
    """Learnable scalar applied after Conv_Reg at each pyramid level."""

    def __init__(self, init_value: float = 1.0) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(init_value)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class ConvGN(nn.Module):
    """3x3 convolution + GroupNorm + SiLU (Conv_GN in Figure 5)."""

    def __init__(self, c1: int, c2: int, groups: int = 8) -> None:
        super().__init__()
        groups = min(groups, c2)
        while c2 % groups:
            groups -= 1
        self.conv = nn.Conv2d(c1, c2, 3, padding=1, bias=False)
        self.gn = nn.GroupNorm(groups, c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.gn(self.conv(x)))


class LayerAttention(nn.Module):
    """Task-specific attention from global average pooling."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, channels, 1)
        self.act = nn.ReLU(inplace=True)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.avg(x)
        w = self.fc2(self.act(self.fc1(w)))
        return self.gate(w)


class TaskDecomposition(nn.Module):
    """Create separate regression and classification representations."""

    def __init__(self, channels: int, out_channels: int) -> None:
        super().__init__()
        self.reg_attn = LayerAttention(channels)
        self.cls_attn = LayerAttention(channels)
        self.reg_proj = nn.Conv2d(channels, out_channels, 3, padding=1, bias=False)
        self.cls_proj = nn.Conv2d(channels, out_channels, 3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        reg = self.reg_proj(x * self.reg_attn(x))
        cls = self.cls_proj(x * self.cls_attn(x))
        return reg, cls


class CPEM(nn.Module):
    """Classification Probability Estimation Module."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = max(channels // 4, 8)
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DCNv3Lite(nn.Module):
    """Portable modulated deformable-convolution approximation of the DCNv3 stage.

    The manuscript specifies DCNv3 but does not expose every implementation-level
    hyperparameter. This adapter preserves learned offsets and modulation while avoiding
    a third-party compiled extension. Replace it with the exact original DCNv3 operator
    before claiming byte-for-byte source equivalence.
    """

    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        k2 = kernel_size * kernel_size
        self.offset_mask = nn.Conv2d(channels, 3 * k2, 3, padding=1)
        self.weight = nn.Parameter(torch.empty(channels, channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(channels))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        om = self.offset_mask(x)
        k2 = self.kernel_size * self.kernel_size
        offset = om[:, : 2 * k2]
        mask = om[:, 2 * k2 :].sigmoid()
        return deform_conv2d(
            x,
            offset,
            self.weight,
            self.bias,
            stride=(1, 1),
            padding=(self.kernel_size // 2, self.kernel_size // 2),
            dilation=(1, 1),
            mask=mask,
        )


class TFE(nn.Module):
    """Multi-receptive-field enhancement accompanying SSFF."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.b3 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.b5 = nn.Conv2d(channels, channels, 5, padding=2, groups=channels, bias=False)
        self.b7 = nn.Conv2d(channels, channels, 7, padding=3, groups=channels, bias=False)
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fuse(torch.cat((self.b3(x), self.b5(x), self.b7(x)), dim=1))


class ASF(nn.Module):
    """Attentional Scale Sequence Fusion for four P2-P5 inputs."""

    def __init__(self, in_channels: Sequence[int], out_channels: int = 64) -> None:
        super().__init__()
        if len(in_channels) != 4:
            raise ValueError(f"ASF expects four inputs, got {len(in_channels)}")
        self.proj = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(c, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(inplace=True),
            )
            for c in in_channels
        )
        self.scale_fuse = nn.ModuleList(
            nn.Sequential(
                nn.Conv3d(out_channels, out_channels, kernel_size=(4, 1, 1), bias=False),
                nn.BatchNorm3d(out_channels),
                nn.SiLU(inplace=True),
            )
            for _ in range(4)
        )
        self.tfe = nn.ModuleList(TFE(out_channels) for _ in range(4))
        self.out = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(inplace=True),
            )
            for _ in range(4)
        )

    def forward(self, xs: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        projected = [layer(x) for layer, x in zip(self.proj, xs)]
        outputs = []
        for i, target in enumerate(projected):
            size = target.shape[-2:]
            aligned = [
                p if p.shape[-2:] == size
                else F.interpolate(p, size=size, mode="bilinear", align_corners=False)
                for p in projected
            ]
            seq = torch.stack(aligned, dim=2)  # B,C,4,H,W
            ssff = self.scale_fuse[i](seq).squeeze(2)
            outputs.append(self.out[i](ssff + self.tfe[i](target)))
        return outputs


class LEGM(nn.Module):
    """Local Enhancement-Global Modulation reconstructed from the manuscript."""

    def __init__(self, channels: int, key_dim: int | None = None) -> None:
        super().__init__()
        key_dim = key_dim or max(channels // 8, 16)
        self.local = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )
        self.q = nn.Conv2d(channels, key_dim, 1, bias=False)
        self.k = nn.Conv2d(channels, key_dim, 1, bias=False)
        self.v = nn.Conv2d(channels, channels, 1, bias=False)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 1),
            nn.GELU(),
            nn.Conv2d(channels * 2, channels, 1),
        )
        self.norm = nn.GroupNorm(1, channels)
        self.out = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.scale = key_dim**-0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fin = self.local(x)
        b, c, h, w = fin.shape
        q = self.q(fin).flatten(2).transpose(1, 2)
        k = self.k(fin).flatten(2)
        v = self.v(fin).flatten(2).transpose(1, 2)
        attn = torch.softmax(torch.bmm(q, k) * self.scale, dim=-1)
        fatt = torch.bmm(attn, v).transpose(1, 2).reshape(b, c, h, w)
        return self.out(self.norm(fin + self.mlp(fatt)))


class C3k2LEGM(nn.Module):
    """Preserve a YOLO11 C3k2 path, then apply LEGM modulation."""

    def __init__(self, c3k2: nn.Module, channels: int) -> None:
        super().__init__()
        self.c3k2 = c3k2
        self.legm = LEGM(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.legm(self.c3k2(x))


class ASFDetect(Detect):
    """Stock Ultralytics Detect head preceded by ASF."""

    def __init__(self, nc: int, ch: Sequence[int], reg_max: int = 16, out_channels: int = 64) -> None:
        self.input_channels = tuple(int(c) for c in ch)
        super().__init__(nc=nc, reg_max=reg_max, end2end=False, ch=(out_channels,) * 4)
        self.asf = ASF(self.input_channels, out_channels)

    def forward(self, x):
        return super().forward(self.asf(x))


class DAHDetect(Detect):
    """Four-level Dynamic Aligned Head compatible with Ultralytics loss/inference."""

    def __init__(
        self,
        nc: int,
        ch: Sequence[int],
        reg_max: int = 16,
        use_asf: bool = False,
        asf_channels: int = 64,
        hidden_channels: int = 64,
    ) -> None:
        self.input_channels = tuple(int(c) for c in ch)
        self.use_asf = bool(use_asf)
        proc_channels = (asf_channels,) * 4 if use_asf else self.input_channels
        super().__init__(nc=nc, reg_max=reg_max, end2end=False, ch=proc_channels)
        self.cv2 = None
        self.cv3 = None

        self.asf = ASF(self.input_channels, asf_channels) if use_asf else nn.Identity()
        hidden = [min(hidden_channels, max(32, c)) for c in proc_channels]
        self.shared1 = nn.ModuleList(ConvGN(c, h) for c, h in zip(proc_channels, hidden))
        self.shared2 = nn.ModuleList(ConvGN(h, h) for h in hidden)
        self.td = nn.ModuleList(TaskDecomposition(2 * h, h) for h in hidden)
        self.dcn = nn.ModuleList(DCNv3Lite(h) for h in hidden)
        self.cpem = nn.ModuleList(CPEM(2 * h) for h in hidden)
        self.reg_out = nn.ModuleList(nn.Conv2d(h, 4 * reg_max, 1) for h in hidden)
        self.cls_out = nn.ModuleList(nn.Conv2d(h, nc, 1) for h in hidden)
        self.scales = nn.ModuleList(Scale(1.0) for _ in hidden)

    def _features(self, x):
        return self.asf(x) if self.use_asf else x

    def forward(self, x):
        feats = list(self._features(x))
        boxes, scores = [], []
        bs = feats[0].shape[0]

        for i, feat in enumerate(feats):
            s1 = self.shared1[i](feat)
            s2 = self.shared2[i](s1)
            inter = torch.cat((s1, s2), dim=1)
            reg_feat, cls_feat = self.td[i](inter)
            reg_feat = self.dcn[i](reg_feat)
            box = self.scales[i](self.reg_out[i](reg_feat))
            prob = self.cpem[i](inter)
            cls = self.cls_out[i](cls_feat * prob)
            boxes.append(box.view(bs, 4 * self.reg_max, -1))
            scores.append(cls.view(bs, self.nc, -1))

        preds = {
            "boxes": torch.cat(boxes, dim=-1),
            "scores": torch.cat(scores, dim=-1),
            "feats": feats,
        }
        if self.training:
            return preds
        y = self._inference(preds)
        return y if self.export else (y, preds)

    def bias_init(self) -> None:
        if not torch.all(self.stride > 0):
            return
        for reg, cls, s in zip(self.reg_out, self.cls_out, self.stride):
            nn.init.constant_(reg.bias, 1.0)
            nn.init.constant_(cls.bias, math.log(5 / self.nc / (640 / float(s)) ** 2))


def infer_detect_input_channels(head: Detect) -> list[int]:
    channels = []
    for branch in head.cv2:
        conv = getattr(branch[0], "conv", None)
        if conv is None:
            raise RuntimeError("Unsupported Ultralytics Detect layout")
        channels.append(int(conv.in_channels))
    return channels


def infer_module_out_channels(module: nn.Module) -> int:
    for attr in ("cv2", "cv3", "cv1"):
        block = getattr(module, attr, None)
        conv = getattr(block, "conv", None)
        if conv is not None:
            return int(conv.out_channels)
    for sub in reversed(list(module.modules())):
        if isinstance(sub, nn.Conv2d):
            return int(sub.out_channels)
    raise RuntimeError(f"Unable to infer output channels for {type(module).__name__}")
