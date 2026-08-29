import pytest
import torch

from aegisfold.models.fusion import Aegis


def test_default_model_returns_one_logit_per_protein() -> None:
    model = Aegis()
    sequence_embeddings = torch.randn(8, 320)
    structure_embeddings = torch.randn(8, 512)

    logits = model(sequence_embeddings, structure_embeddings)

    assert logits.shape == (8,)


def test_model_supports_configurable_embedding_dimensions() -> None:
    model = Aegis(
        sequence_dim=16,
        structure_dim=24,
        projection_dim=8,
        hidden_dim=4,
    )
    sequence_embeddings = torch.randn(3, 16)
    structure_embeddings = torch.randn(3, 24)

    logits = model(sequence_embeddings, structure_embeddings)

    assert logits.shape == (3,)


@pytest.mark.parametrize(
    ("sequence_shape", "structure_shape", "message"),
    [
        ((320,), (1, 512), "sequence_embedding"),
        ((1, 320), (512,), "structure_embedding"),
    ],
)
def test_model_rejects_unbatched_embeddings(
    sequence_shape: tuple[int, ...],
    structure_shape: tuple[int, ...],
    message: str,
) -> None:
    model = Aegis()

    with pytest.raises(ValueError, match=message):
        model(
            torch.randn(sequence_shape),
            torch.randn(structure_shape),
        )


def test_model_rejects_mismatched_batch_sizes() -> None:
    model = Aegis()

    with pytest.raises(ValueError, match="batch sizes must match"):
        model(
            torch.randn(4, 320),
            torch.randn(3, 512),
        )


def test_all_trainable_parameters_receive_gradients() -> None:
    model = Aegis()
    sequence_embeddings = torch.randn(4, 320)
    structure_embeddings = torch.randn(4, 512)
    labels = torch.tensor([0.0, 1.0, 0.0, 1.0])

    logits = model(sequence_embeddings, structure_embeddings)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss.backward()

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    assert trainable_parameters
    assert all(parameter.grad is not None for parameter in trainable_parameters)


def test_evaluation_mode_is_deterministic() -> None:
    torch.manual_seed(7)
    model = Aegis(dropout=0.5).eval()
    sequence_embeddings = torch.randn(2, 320)
    structure_embeddings = torch.randn(2, 512)

    with torch.inference_mode():
        first = model(sequence_embeddings, structure_embeddings)
        second = model(sequence_embeddings, structure_embeddings)

    assert torch.equal(first, second)
