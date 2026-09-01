from dataclasses import dataclass

import torch
from torch import nn


def _validate_feature_matrix(
    value: torch.Tensor,
    *,
    name: str,
    expected_width: int,
    expected_batch_size: int | None = None,
) -> None:
    """Validate one batched feature matrix before it reaches a linear layer."""

    if value.ndim != 2:
        raise ValueError(
            f"{name} must have shape [batch, {expected_width}], got {tuple(value.shape)}"
        )
    if value.shape[1] != expected_width:
        raise ValueError(f"{name} must have feature width {expected_width}, got {value.shape[1]}")
    if expected_batch_size is not None and value.shape[0] != expected_batch_size:
        raise ValueError(
            "all Aegis inputs must have the same batch size; "
            f"{name} has {value.shape[0]}, expected {expected_batch_size}"
        )


@dataclass(frozen=True, slots=True)
class AegisOutput:
    """Raw logits emitted by the complete model and its two pathways."""

    overall_logit: torch.Tensor
    embedding_logit: torch.Tensor
    reference_logit: torch.Tensor

    @property
    def overall_probability(self) -> torch.Tensor:
        """Return the uncalibrated sigmoid of the overall logit."""

        return torch.sigmoid(self.overall_logit)

    @property
    def embedding_probability(self) -> torch.Tensor:
        """Return the uncalibrated sigmoid of the embedding-pathway logit."""

        return torch.sigmoid(self.embedding_logit)

    @property
    def reference_probability(self) -> torch.Tensor:
        """Return the uncalibrated sigmoid of the reference-pathway logit."""

        return torch.sigmoid(self.reference_logit)


class EmbeddingFusionEncoder(nn.Module):
    """Fuse pooled ESM-2 and ESM-IF1 representations into one vector."""

    def __init__(
        self,
        sequence_dim: int = 320,
        structure_dim: int = 512,
        projection_dim: int = 128,
        output_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.sequence_dim = sequence_dim
        self.structure_dim = structure_dim
        self.output_dim = output_dim

        self.sequence_projector = nn.Sequential(
            nn.Linear(sequence_dim, projection_dim),
            nn.GELU(),
            nn.LayerNorm(projection_dim),
            nn.Dropout(dropout),
        )
        self.structure_projector = nn.Sequential(
            nn.Linear(structure_dim, projection_dim),
            nn.GELU(),
            nn.LayerNorm(projection_dim),
            nn.Dropout(dropout),
        )
        self.fusion = nn.Sequential(
            nn.Linear(projection_dim * 2, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        sequence_embedding: torch.Tensor,
        structure_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Return an embedding-derived representation for each protein."""

        _validate_feature_matrix(
            sequence_embedding,
            name="sequence_embedding",
            expected_width=self.sequence_dim,
        )
        _validate_feature_matrix(
            structure_embedding,
            name="structure_embedding",
            expected_width=self.structure_dim,
            expected_batch_size=sequence_embedding.shape[0],
        )

        sequence_projection = self.sequence_projector(sequence_embedding)
        structure_projection = self.structure_projector(structure_embedding)
        fused = torch.cat([sequence_projection, structure_projection], dim=-1)
        return self.fusion(fused)


class ReferenceEvidenceEncoder(nn.Module):
    """Fuse reference-similarity measurements and quality/context features."""

    def __init__(
        self,
        sequence_feature_dim: int = 5,
        structure_feature_dim: int = 5,
        context_dim: int = 5,
        projection_dim: int = 16,
        output_dim: int = 32,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.sequence_feature_dim = sequence_feature_dim
        self.structure_feature_dim = structure_feature_dim
        self.context_dim = context_dim
        self.output_dim = output_dim

        self.sequence_projector = nn.Sequential(
            nn.Linear(sequence_feature_dim, projection_dim),
            nn.GELU(),
            nn.LayerNorm(projection_dim),
            nn.Dropout(dropout),
        )
        self.structure_projector = nn.Sequential(
            nn.Linear(structure_feature_dim, projection_dim),
            nn.GELU(),
            nn.LayerNorm(projection_dim),
            nn.Dropout(dropout),
        )
        self.fusion = nn.Sequential(
            nn.Linear(projection_dim * 2 + context_dim, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        sequence_evidence: torch.Tensor,
        structure_evidence: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """Return a reference-derived representation for each protein."""

        _validate_feature_matrix(
            sequence_evidence,
            name="sequence_evidence",
            expected_width=self.sequence_feature_dim,
        )
        batch_size = sequence_evidence.shape[0]
        _validate_feature_matrix(
            structure_evidence,
            name="structure_evidence",
            expected_width=self.structure_feature_dim,
            expected_batch_size=batch_size,
        )
        _validate_feature_matrix(
            context,
            name="context",
            expected_width=self.context_dim,
            expected_batch_size=batch_size,
        )

        sequence_projection = self.sequence_projector(sequence_evidence)
        structure_projection = self.structure_projector(structure_evidence)
        fused = torch.cat([sequence_projection, structure_projection, context], dim=-1)
        return self.fusion(fused)


class Aegis(nn.Module):
    """Combine embedding fusion and reference-evidence fusion for screening."""

    def __init__(
        self,
        sequence_dim: int = 320,
        structure_dim: int = 512,
        sequence_feature_dim: int = 5,
        structure_feature_dim: int = 5,
        context_dim: int = 5,
        embedding_projection_dim: int = 128,
        embedding_output_dim: int = 128,
        reference_projection_dim: int = 16,
        reference_output_dim: int = 32,
        final_hidden_dim: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.embedding_encoder = EmbeddingFusionEncoder(
            sequence_dim=sequence_dim,
            structure_dim=structure_dim,
            projection_dim=embedding_projection_dim,
            output_dim=embedding_output_dim,
            dropout=dropout,
        )
        self.reference_encoder = ReferenceEvidenceEncoder(
            sequence_feature_dim=sequence_feature_dim,
            structure_feature_dim=structure_feature_dim,
            context_dim=context_dim,
            projection_dim=reference_projection_dim,
            output_dim=reference_output_dim,
            dropout=dropout,
        )
        self.embedding_head = nn.Linear(embedding_output_dim, 1)
        self.reference_head = nn.Linear(reference_output_dim, 1)
        self.final_head = nn.Sequential(
            nn.Linear(embedding_output_dim + reference_output_dim, final_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(final_hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(final_hidden_dim, 1),
        )

    def forward(
        self,
        sequence_embedding: torch.Tensor,
        structure_embedding: torch.Tensor,
        sequence_evidence: torch.Tensor,
        structure_evidence: torch.Tensor,
        context: torch.Tensor,
    ) -> AegisOutput:
        """Return overall and pathway-specific logits for a batch of proteins."""

        embedding_representation = self.embedding_encoder(
            sequence_embedding,
            structure_embedding,
        )
        reference_representation = self.reference_encoder(
            sequence_evidence,
            structure_evidence,
            context,
        )
        if embedding_representation.shape[0] != reference_representation.shape[0]:
            raise ValueError(
                "all Aegis inputs must have the same batch size; embedding inputs have "
                f"{embedding_representation.shape[0]}, reference inputs have "
                f"{reference_representation.shape[0]}"
            )

        combined = torch.cat(
            [embedding_representation, reference_representation],
            dim=-1,
        )
        return AegisOutput(
            overall_logit=self.final_head(combined).squeeze(-1),
            embedding_logit=self.embedding_head(embedding_representation).squeeze(-1),
            reference_logit=self.reference_head(reference_representation).squeeze(-1),
        )
