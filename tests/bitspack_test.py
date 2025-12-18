"""
Test module for bitspack.
"""

import torch
import qmp.bitspack

# pylint: disable=missing-function-docstring


def test_pack_unpack_int_size_1() -> None:
    tensor = torch.tensor([1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1], dtype=torch.uint8)
    size = 1
    packed = qmp.bitspack.pack_int(tensor, size)
    unpacked = qmp.bitspack.unpack_int(packed, size, tensor.shape[-1])
    assert torch.all(unpacked == tensor)


def test_pack_unpack_int_size_2() -> None:
    tensor = torch.tensor([3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0], dtype=torch.uint8)
    size = 2
    packed = qmp.bitspack.pack_int(tensor, size)
    unpacked = qmp.bitspack.unpack_int(packed, size, tensor.shape[-1])
    assert torch.all(unpacked == tensor)


def test_pack_unpack_int_size_4() -> None:
    tensor = torch.tensor([15, 10, 5, 0, 15, 10, 5, 0, 15, 10, 5, 0, 15, 10, 5, 0], dtype=torch.uint8)
    size = 4
    packed = qmp.bitspack.pack_int(tensor, size)
    unpacked = qmp.bitspack.unpack_int(packed, size, tensor.shape[-1])
    assert torch.all(unpacked == tensor)


def test_pack_unpack_int_size_8() -> None:
    tensor = torch.tensor([15, 210, 5, 123, 151, 130, 5, 0, 15, 10, 75, 0, 15, 10, 5, 0], dtype=torch.uint8)
    size = 8
    packed = qmp.bitspack.pack_int(tensor, size)
    unpacked = qmp.bitspack.unpack_int(packed, size, tensor.shape[-1])
    assert torch.all(unpacked == tensor)


def test_pack_unpack_int_with_padding() -> None:
    tensor = torch.tensor([1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0], dtype=torch.uint8)
    size = 1
    packed = qmp.bitspack.pack_int(tensor, size)
    unpacked = qmp.bitspack.unpack_int(packed, size, tensor.shape[-1])
    assert torch.all(unpacked == tensor)


def test_pack_unpack_int_multi_dimensional() -> None:
    tensor = torch.tensor([[1, 0, 1, 1], [0, 0, 1, 0], [1, 1, 0, 1], [1, 0, 0, 1]], dtype=torch.uint8)
    size = 1
    packed = qmp.bitspack.pack_int(tensor, size)
    unpacked = qmp.bitspack.unpack_int(packed, size, tensor.shape[-1])
    assert torch.all(unpacked == tensor)


def test_pack_unpack_int_size_2_with_padding() -> None:
    tensor = torch.tensor([3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2], dtype=torch.uint8)
    size = 2
    packed = qmp.bitspack.pack_int(tensor, size)
    unpacked = qmp.bitspack.unpack_int(packed, size, tensor.shape[-1])
    assert torch.all(unpacked == tensor)


def test_pack_unpack_int_size_4_with_padding() -> None:
    tensor = torch.tensor([15, 10, 5, 0, 15, 10, 5, 0, 15, 10, 5, 0, 15, 10], dtype=torch.uint8)
    size = 4
    packed = qmp.bitspack.pack_int(tensor, size)
    unpacked = qmp.bitspack.unpack_int(packed, size, tensor.shape[-1])
    assert torch.all(unpacked == tensor)


def test_pack_unpack_int_size_1_large_tensor() -> None:
    tensor = torch.randint(0, 2, (1000,), dtype=torch.uint8)
    size = 1
    packed = qmp.bitspack.pack_int(tensor, size)
    unpacked = qmp.bitspack.unpack_int(packed, size, tensor.shape[-1])
    assert torch.all(unpacked == tensor)


def test_pack_unpack_int_size_2_large_tensor() -> None:
    tensor = torch.randint(0, 4, (1000,), dtype=torch.uint8)
    size = 2
    packed = qmp.bitspack.pack_int(tensor, size)
    unpacked = qmp.bitspack.unpack_int(packed, size, tensor.shape[-1])
    assert torch.all(unpacked == tensor)


def test_pack_unpack_int_size_4_large_tensor() -> None:
    tensor = torch.randint(0, 16, (1000,), dtype=torch.uint8)
    size = 4
    packed = qmp.bitspack.pack_int(tensor, size)
    unpacked = qmp.bitspack.unpack_int(packed, size, tensor.shape[-1])
    assert torch.all(unpacked == tensor)
