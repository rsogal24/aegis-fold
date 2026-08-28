import torch

from aegisfold.embeddings.base import EmbeddingResult
from aegisfold.io import save_embedding


def test_save_embedding(tmp_path) -> None:
    result = EmbeddingResult(
        vector=torch.tensor([1.0, 2.0]),
        model_name="test-model",
        residue_count=4,
    )

    output = save_embedding(result, tmp_path / "nested" / "embedding.pt")
    saved = torch.load(output, weights_only=True)

    assert saved["dimension"] == 2
    assert saved["residue_count"] == 4
    assert torch.equal(saved["embedding"], result.vector)
