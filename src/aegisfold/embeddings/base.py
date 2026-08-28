"""Shared embedding types and pooling operations."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    """A fixed-size protein representation with provenance metadata."""

    vector: torch.Tensor
    model_name: str
    residue_count: int
    pooling: str = "mean"

    @property
    def dimension(self) -> int:
        return int(self.vector.shape[-1])


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool residue representations selected by a one-dimensional mask."""

    if values.ndim != 2:
        raise ValueError("values must have shape [residues, features]")
    if mask.ndim != 1 or mask.shape[0] != values.shape[0]:
        raise ValueError("mask must have shape [residues]")

    selected = values[mask.to(device=values.device, dtype=torch.bool)]
    if selected.shape[0] == 0:
        raise ValueError("cannot pool an embedding with no valid residues")
    return selected.mean(dim=0)

