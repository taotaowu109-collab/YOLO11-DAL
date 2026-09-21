# YOLO11-DAL

Implementation of:

**YOLO11-DAL: A multi-scale fusion network for target detection in UAV-based power line inspection**

Authors: Taotao Wu, Zhonghua Liu.

> **Status**
>
> This repository provides the implementation, configuration files, and reproduction instructions for YOLO11-DAL.

## Implemented

- YOLO11n baseline.
- P2-P5 control model for the reviewer-requested `YOLO11n + P2 + original head` experiment.
- C3k2-LEGM.
- ASF with P2-P5 alignment, scale stacking, Conv3D SSFF path and TFE path.
- Four-level DAH with Conv_GN, TD, LayerAttention, CPEM, spatially adaptive regression, Conv_Reg -> Scale and Conv_Cls.
- Complete 2 x 2 x 2 ablation variants plus P2-only control.
- Train, validation, prediction, model-info and smoke-test scripts.
- Reference CSV containing the manuscript and response-letter values.

See [`docs/MAPPING_TO_MANUSCRIPT.md`](docs/MAPPING_TO_MANUSCRIPT.md) for the manuscript-to-code mapping and implementation details to verify.

## Layout

```text
configs/
  data/powerline.yaml
  models/yolo11n_baseline.yaml
  models/yolo11n_p2.yaml
docs/MAPPING_TO_MANUSCRIPT.md
reference/reported_metrics.csv
scripts/
  ablation.py
  model_info.py
  predict.py
  smoke_test.py
  train.py
  val.py
tests/test_modules.py
yolo11_dal/
  __init__.py
  model.py
  modules.py
```

## Environment

The current manuscript reports:

- Windows 10
- NVIDIA RTX 3090
- CUDA 12.1
- OpenCV 3.4.6
- 640 x 640 input
- 200 epochs
- batch size 16
- initial learning rate 0.01
- momentum 0.937
- weight decay 0.0005

The current codebase is pinned to `ultralytics==8.4.157` because its internal API is used by the variant builder.

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

python -m pip install --upgrade pip
pip install -e .
```

The augmentation settings, seed and DCNv3 operator/configuration should be checked against the experiment setup before exact-reproduction claims are made.

## Dataset

The manuscript uses the public seven-class `BK201seven/613e` UAV power-line dataset:

- 7,612 images
- 5,327 train
- 1,523 validation
- 762 test

Classes:

1. insulator
2. insulator_stringdrop
3. insulator_breakage
4. insulator_flashover
5. damper
6. damper_defect
7. nest

Expected local layout:

```text
datasets/613e/
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
```

Edit `configs/data/powerline.yaml` if the local path differs.

## Quick check

```bash
python scripts/smoke_test.py --variant full --imgsz 640 --device 0
python scripts/model_info.py --all
pytest -q
```

## Train full model

```bash
python scripts/train.py ^
  --variant full ^
  --data configs/data/powerline.yaml ^
  --epochs 200 ^
  --batch 16 ^
  --imgsz 640 ^
  --lr0 0.01 ^
  --momentum 0.937 ^
  --weight-decay 0.0005 ^
  --optimizer SGD ^
  --device 0
```

## Ablation variants

| Variant | P2 | C3k2-LEGM | ASF | DAH |
|---|:---:|:---:|:---:|:---:|
| `baseline` |  |  |  |  |
| `p2` | yes |  |  |  |
| `dah` | yes |  |  | yes |
| `legm` |  | yes |  |  |
| `asf` | yes |  | yes |  |
| `dah_legm` | yes | yes |  | yes |
| `dah_asf` | yes |  | yes | yes |
| `legm_asf` | yes | yes | yes |  |
| `full` | yes | yes | yes | yes |

Run the manuscript 2 x 2 x 2 ablation:

```bash
python scripts/ablation.py --data configs/data/powerline.yaml --device 0
```

Add the reviewer P2-only control:

```bash
python scripts/ablation.py --data configs/data/powerline.yaml --device 0 --include-p2-control
```

## Reported values

The values below are copied from the manuscript/response letter into `reference/reported_metrics.csv` for verification.

| Variant | Params M | GFLOPs | P % | R % | mAP50 % | mAP50:95 % |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 2.58 | 6.3 | 87.7 | 75.9 | 82.9 | 55.2 |
| P2-only | 2.68 | 8.7 | 88.1 | 76.5 | 83.4 | 56.5 |
| DAH | 2.08 | 5.4 | 88.6 | 77.5 | 84.6 | 57.5 |
| full | 2.34 | 6.1 | 89.2 | 80.1 | 86.2 | 57.7 |

## DCNv3 implementation note

The current DAH code includes a spatially adaptive deformable-convolution operator with learned offsets and masks. Its exact settings should be checked against the configuration used for the reported experiment before the final public/DOI release.

## Before making the repository public or creating a DOI archive

1. Run all smoke tests.
2. Train baseline, P2-only, DAH-only and full variants.
3. Compare parameters, GFLOPs and metrics against `reference/reported_metrics.csv`.
4. Resolve the implementation details listed in `docs/MAPPING_TO_MANUSCRIPT.md`.
5. Record exact Python/PyTorch/CUDA/Ultralytics versions, seed and augmentation settings.
6. Commit verified code and create an immutable release tag.
7. Archive the verified release if a DOI is required.

## Third-party software

This project uses Ultralytics YOLO. Review and comply with upstream licensing terms before public redistribution.
