"""Persistence helpers for generated representations."""

from pathlib import Path

import torch

from aegisfold.embeddings.base import EmbeddingResult


def save_embedding(result: EmbeddingResult, output_path: str | Path) -> Path:
    """Save an embedding and its provenance as a PyTorch artifact."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "embedding": result.vector,
            "model_name": result.model_name,
            "residue_count": result.residue_count,
            "pooling": result.pooling,
            "dimension": result.dimension,
        },
        path,
    )
    return path

