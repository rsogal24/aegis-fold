"""Trainable AegisFold model architectures."""

from aegisfold.models.fusion import (
    Aegis,
    AegisOutput,
    EmbeddingFusionEncoder,
    ReferenceEvidenceEncoder,
)

__all__ = [
    "Aegis",
    "AegisOutput",
    "EmbeddingFusionEncoder",
    "ReferenceEvidenceEncoder",
]
