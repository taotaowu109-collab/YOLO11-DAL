# Mapping from manuscript to code

This document maps the manuscript components to the current codebase.

| Manuscript item | Code |
|---|---|
| C3k2-LEGM | `yolo11_dal.modules.C3k2LEGM` + `LEGM` |
| ASF | `yolo11_dal.modules.ASF` |
| SSFF scale stacking + Conv3D | `ASF.forward()` + `scale_fuse` |
| TFE | `yolo11_dal.modules.TFE` |
| Conv_GN | `yolo11_dal.modules.ConvGN` |
| TD / LayerAttention | `TaskDecomposition` + `LayerAttention` |
| CPEM | `yolo11_dal.modules.CPEM` |
| DCNv3 stage | `DCNv3Lite` |
| Conv_Reg -> Scale | `DAHDetect.reg_out` -> `Scale` |
| Conv_Cls | `DAHDetect.cls_out` |
| P2-P5 DAH | `DAHDetect` on four pyramid inputs |
| 2x2x2 ablation | `scripts/ablation.py` |
| P2-only reviewer control | variant `p2` |

## Implementation details to verify before final release

1. Verify the exact DCNv3 operator/configuration used for the reported experiment.
2. Verify the two C3k2-LEGM insertion positions against the experiment code.
3. Verify the ASF channel settings and P2-P5 fusion configuration.
4. Verify the P2-only control architecture used for the reviewer experiment.
5. Verify optimizer, augmentation, random seed and all environment versions used for the reported runs.
6. Confirm that model parameters, GFLOPs and evaluation metrics match the manuscript tables before public release.

After verification, update this document and tag the exact release used for the paper.
