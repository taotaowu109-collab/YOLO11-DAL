# Mapping from manuscript to code

This repository is a reconstruction from the current manuscript and its cited source designs.

| Manuscript item | Code |
|---|---|
| C3k2-LEGM | `yolo11_dal.modules.C3k2LEGM` + `LEGM` |
| ASF | `yolo11_dal.modules.ASF` |
| SSFF scale stacking + Conv3D | `ASF.forward()` + `scale_fuse` |
| TFE | `yolo11_dal.modules.TFE` |
| Conv_GN | `yolo11_dal.modules.ConvGN` |
| TD / LayerAttention | `TaskDecomposition` + `LayerAttention` |
| CPEM | `yolo11_dal.modules.CPEM` |
| DCNv3 stage | `DCNv3Lite` (portable reconstruction) |
| Conv_Reg -> Scale | `DAHDetect.reg_out` -> `Scale` |
| Conv_Cls | `DAHDetect.cls_out` |
| P2-P5 DAH | `DAHDetect` on four pyramid inputs |
| 2x2x2 ablation | `scripts/ablation.py` |
| P2-only reviewer control | variant `p2` |

## Reconstruction choices that must be verified

The manuscript does not expose every implementation-level choice needed to recreate the original training repository byte-for-byte. The following are explicit reconstruction choices, not claims about unpublished source:

1. `DCNv3Lite` uses `torchvision.ops.deform_conv2d` with learned offsets and a modulation mask. Replace it with the exact DCNv3 operator used in the original run if available.
2. The two C3k2-LEGM replacements are placed at backbone indices 6 and 8.
3. ASF is applied to the four P2-P5 neck outputs immediately before the prediction head.
4. The P2-only skeleton extends the YOLO11n FPN/PAN pattern with an additional P2 branch.
5. The manuscript specifies image size, epochs, batch size, initial learning rate, momentum and weight decay, but does not expose every optimizer/augmentation/random-seed detail. The training CLI keeps these choices visible.

Do not describe this repository as recovered original author source until it has been checked against the original training files. After verification, update this document and tag the exact release used for the paper.
