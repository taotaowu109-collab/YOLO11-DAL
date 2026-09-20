import torch

from yolo11_dal.modules import ASF, CPEM, TaskDecomposition


def test_asf_shapes():
    m = ASF([64, 64, 128, 256], out_channels=32)
    xs = [
        torch.randn(1, 64, 32, 32),
        torch.randn(1, 64, 16, 16),
        torch.randn(1, 128, 8, 8),
        torch.randn(1, 256, 4, 4),
    ]
    ys = m(xs)
    assert [y.shape for y in ys] == [
        (1, 32, 32, 32),
        (1, 32, 16, 16),
        (1, 32, 8, 8),
        (1, 32, 4, 4),
    ]


def test_task_decomposition_shapes():
    m = TaskDecomposition(64, 32)
    reg, cls = m(torch.randn(2, 64, 20, 20))
    assert reg.shape == cls.shape == (2, 32, 20, 20)


def test_cpem_shape():
    m = CPEM(64)
    y = m(torch.randn(2, 64, 20, 20))
    assert y.shape == (2, 1, 20, 20)
