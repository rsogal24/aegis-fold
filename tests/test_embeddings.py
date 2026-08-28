import pytest
import torch

from aegisfold.embeddings.base import masked_mean
from aegisfold.embeddings.sequence import normalize_sequence


def test_normalize_sequence() -> None:
    assert normalize_sequence(" acd\nEFG ") == "ACDEFG"


def test_normalize_sequence_rejects_invalid_characters() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        normalize_sequence("ACD*")


def test_masked_mean() -> None:
    values = torch.tensor([[1.0, 2.0], [3.0, 4.0], [9.0, 10.0]])
    mask = torch.tensor([True, True, False])

    result = masked_mean(values, mask)

    assert torch.equal(result, torch.tensor([2.0, 3.0]))

