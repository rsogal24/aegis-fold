import pytest
import torch

from aegisfold.models.fusion import (
    Aegis,
    AegisOutput,
    EmbeddingFusionEncoder,
    ReferenceEvidenceEncoder,
)


def make_default_inputs(batch_size: int = 8) -> tuple[torch.Tensor, ...]:
    return (
        torch.randn(batch_size, 320),
        torch.randn(batch_size, 512),
        torch.randn(batch_size, 5),
        torch.randn(batch_size, 5),
        torch.randn(batch_size, 5),
    )


def test_embedding_encoder_returns_configured_representation() -> None:
    encoder = EmbeddingFusionEncoder(output_dim=48)

    result = encoder(torch.randn(3, 320), torch.randn(3, 512))

    assert result.shape == (3, 48)


def test_reference_encoder_returns_configured_representation() -> None:
    encoder = ReferenceEvidenceEncoder(output_dim=24)

    result = encoder(
        torch.randn(3, 5),
        torch.randn(3, 5),
        torch.randn(3, 5),
    )

    assert result.shape == (3, 24)


def test_aegis_returns_three_logits_per_protein() -> None:
    output = Aegis()(*make_default_inputs())

    assert isinstance(output, AegisOutput)
    assert output.overall_logit.shape == (8,)
    assert output.embedding_logit.shape == (8,)
    assert output.reference_logit.shape == (8,)


def test_aegis_supports_configurable_dimensions() -> None:
    model = Aegis(
        sequence_dim=16,
        structure_dim=24,
        sequence_feature_dim=3,
        structure_feature_dim=4,
        context_dim=2,
        embedding_projection_dim=8,
        embedding_output_dim=6,
        reference_projection_dim=4,
        reference_output_dim=5,
        final_hidden_dim=3,
    )

    output = model(
        torch.randn(2, 16),
        torch.randn(2, 24),
        torch.randn(2, 3),
        torch.randn(2, 4),
        torch.randn(2, 2),
    )

    assert output.overall_logit.shape == (2,)


@pytest.mark.parametrize(
    ("input_index", "bad_shape", "message"),
    [
        (0, (320,), "sequence_embedding"),
        (1, (512,), "structure_embedding"),
        (2, (5,), "sequence_evidence"),
        (3, (5,), "structure_evidence"),
        (4, (5,), "context"),
    ],
)
def test_aegis_rejects_unbatched_inputs(
    input_index: int,
    bad_shape: tuple[int, ...],
    message: str,
) -> None:
    model = Aegis()
    inputs = list(make_default_inputs(batch_size=1))
    inputs[input_index] = torch.randn(bad_shape)

    with pytest.raises(ValueError, match=message):
        model(*inputs)


@pytest.mark.parametrize(
    ("input_index", "bad_width", "message"),
    [
        (0, 319, "sequence_embedding"),
        (1, 511, "structure_embedding"),
        (2, 4, "sequence_evidence"),
        (3, 4, "structure_evidence"),
        (4, 4, "context"),
    ],
)
def test_aegis_rejects_incorrect_feature_widths(
    input_index: int,
    bad_width: int,
    message: str,
) -> None:
    model = Aegis()
    inputs = list(make_default_inputs(batch_size=2))
    inputs[input_index] = torch.randn(2, bad_width)

    with pytest.raises(ValueError, match=message):
        model(*inputs)


def test_aegis_rejects_mismatched_pathway_batch_sizes() -> None:
    model = Aegis()
    inputs = list(make_default_inputs(batch_size=4))
    inputs[2] = torch.randn(3, 5)
    inputs[3] = torch.randn(3, 5)
    inputs[4] = torch.randn(3, 5)

    with pytest.raises(ValueError, match="same batch size"):
        model(*inputs)


def test_combined_auxiliary_loss_reaches_every_trainable_parameter() -> None:
    model = Aegis()
    labels = torch.tensor([0.0, 1.0, 0.0, 1.0])
    output = model(*make_default_inputs(batch_size=4))

    criterion = torch.nn.BCEWithLogitsLoss()
    loss = (
        criterion(output.overall_logit, labels)
        + 0.25 * criterion(output.embedding_logit, labels)
        + 0.25 * criterion(output.reference_logit, labels)
    )
    loss.backward()

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    assert trainable_parameters
    assert all(parameter.grad is not None for parameter in trainable_parameters)


def test_output_probabilities_are_between_zero_and_one() -> None:
    output = Aegis().eval()(*make_default_inputs(batch_size=3))

    probabilities = (
        output.overall_probability,
        output.embedding_probability,
        output.reference_probability,
    )
    assert all(torch.all((value >= 0) & (value <= 1)) for value in probabilities)


def test_evaluation_mode_is_deterministic() -> None:
    torch.manual_seed(7)
    model = Aegis(dropout=0.5).eval()
    inputs = make_default_inputs(batch_size=2)

    with torch.inference_mode():
        first = model(*inputs)
        second = model(*inputs)

    assert torch.equal(first.overall_logit, second.overall_logit)
    assert torch.equal(first.embedding_logit, second.embedding_logit)
    assert torch.equal(first.reference_logit, second.reference_logit)
