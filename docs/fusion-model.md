# Aegis

This guide documents the current model in
`src/aegisfold/models/fusion.py` and its tests in
`tests/test_fusion_model.py`.

The central design decision is that **Aegis combines two different kinds of
screening information**:

1. High-dimensional representations learned by pretrained protein encoders.
2. Explicit evidence produced by comparing a query with monitored and
   background reference proteins.

The model does not receive an amino-acid sequence or PDB file directly. Those
inputs must first pass through the embedding and reference-retrieval pipelines.

## What Aegis predicts

Aegis produces screening logits for a supervised class defined by the training
labels. In the planned public benchmark, the positive class means membership in
a selected benign monitored proxy family, not experimentally established
toxicity.

A high output should eventually mean:

> Under the validated benchmark distribution, the combined embedding and
> reference evidence is sufficiently associated with the monitored class to
> warrant review.

It must not be interpreted as a universal measurement of toxicity. Biological
activity also depends on factors that are not available to this model, and a
predicted structure is derived from the same submitted sequence rather than
being a fully independent observation.

## End-to-end position

```text
                            Query protein
                                 |
             +-------------------+-------------------+
             |                                       |
             v                                       v
      Embedding pipeline                      Reference pipeline
             |                                       |
      +------+------+                         +------+------+
      |             |                         |             |
      v             v                         v             v
 ESM-2 [320]   ESM-IF1 [512]       sequence evidence  structure evidence
      |             |                    [5]               [5]
      +------+------+                         |             |
             |                                +------+------+ 
             v                                       |
  EmbeddingFusionEncoder                    context/quality [5]
             |                                       |
             v                                       v
 embedding representation [128]     ReferenceEvidenceEncoder
             |                                       |
             |                          reference representation [32]
             |                                       |
             +-------------------+-------------------+
                                 |
                                 v
                         final fusion [160]
                                 |
                                 v
                       overall screening logit
```

Reference retrieval, similarity calculation, structure-quality calculation,
feature normalization, probability calibration, and decision thresholds remain
outside this neural module. Keeping these concerns separate makes leakage and
calibration easier to audit.

## Default input contract

For a batch containing `B` query/reference-family pairs, `Aegis.forward()`
expects five floating-point matrices:

| Argument | Default shape | Intended content |
| --- | --- | --- |
| `sequence_embedding` | `[B, 320]` | Pooled ESM-2 query representation |
| `structure_embedding` | `[B, 512]` | Pooled ESM-IF1 query representation |
| `sequence_evidence` | `[B, 5]` | Query-to-reference sequence evidence |
| `structure_evidence` | `[B, 5]` | Query-to-reference structure evidence |
| `context` | `[B, 5]` | Quality, coverage, and OOD information |

All five rows at index `i` must describe the same query and candidate reference
family. Mixing rows between proteins would produce a valid tensor operation but
an invalid biological example, so the future dataset layer must preserve stable
record identifiers in addition to the shape validation implemented here.

The five evidence features are configurable widths. Their exact definitions
must be frozen in the benchmark specification before training. A proposed
version-one definition is:

### Sequence evidence

```text
0  best monitored-reference sequence similarity
1  mean top-k monitored-reference sequence similarity
2  best background-reference sequence similarity
3  monitored-minus-background sequence margin
4  normalized classical sequence-search score
```

### Structure evidence

```text
0  best monitored-reference structure similarity
1  mean top-k monitored-reference structure similarity
2  best background-reference structure similarity
3  monitored-minus-background structure margin
4  normalized structural-search score
```

### Context

```text
0  structure-confidence summary
1  coordinate coverage
2  sequence out-of-distribution distance
3  structure out-of-distribution distance
4  structure-availability indicator
```

These lists describe the intended data contract, not functionality currently
implemented by `fusion.py`. The model consumes numbers; it does not calculate
nearest neighbors or verify that a column contains the claimed measurement.

All evidence and context features should be normalized using parameters fitted
on the training split only. Validation or test statistics must not influence
normalization.

## Model output

The forward method returns an `AegisOutput` object:

```python
@dataclass(frozen=True, slots=True)
class AegisOutput:
    overall_logit: torch.Tensor
    embedding_logit: torch.Tensor
    reference_logit: torch.Tensor
```

For a batch of `B` examples, every tensor has shape `[B]`.

- `overall_logit` is produced from both pathways and is the primary training
  and evaluation output.
- `embedding_logit` measures what the embedding pathway can predict on its own.
- `reference_logit` measures what the explicit reference pathway can predict
  on its own.

The output object also provides convenience properties:

```python
output.overall_probability
output.embedding_probability
output.reference_probability
```

These properties apply `torch.sigmoid()` to the associated logits. They are
**uncalibrated model scores**. They should not be presented as trustworthy
probabilities until a calibration method is fitted on held-out validation data.

Returning all three logits supports scientific ablations and application-level
explanations such as:

```text
Embedding pathway: low evidence
Reference pathway: elevated evidence
Overall pathway:   review threshold exceeded
```

The pathway logits are still learned statistical outputs. Detailed explanations
should also include the actual nearest references, similarity margins, and
structure-quality measurements supplied to the model.

## `_validate_feature_matrix`

The private helper validates every input before it reaches a linear layer. It
checks:

1. The tensor is two-dimensional.
2. Its feature width equals the configured width.
3. Its batch size matches earlier inputs when a batch size is provided.

Example failure:

```text
sequence_embedding must have feature width 320, got 319
```

Explicit validation provides a clearer error than the matrix-multiplication
exception that `nn.Linear` would otherwise raise. It does not currently check
dtype, device, NaN values, or infinite values; the future batch-validation layer
should enforce those properties.

## `EmbeddingFusionEncoder`

This encoder processes high-dimensional query representations from ESM-2 and
ESM-IF1.

Default transformation:

```text
sequence embedding [B, 320]          structure embedding [B, 512]
            |                                    |
      Linear(320, 128)                      Linear(512, 128)
            |                                    |
          GELU                                 GELU
            |                                    |
      LayerNorm(128)                      LayerNorm(128)
            |                                    |
       Dropout(0.2)                        Dropout(0.2)
            |                                    |
            +------------ concatenate -----------+
                              |
                           [B, 256]
                              |
                       Linear(256, 128)
                              |
                            GELU
                              |
                       LayerNorm(128)
                              |
                        Dropout(0.2)
                              |
                 embedding representation [B, 128]
```

The sequence and structure projectors are independent. Equal output widths do
not imply that the original ESM-2 and ESM-IF1 coordinates share meaning. Each
projector learns its own task-specific transformation before concatenation.

The encoder returns a representation rather than a score. `Aegis` attaches an
auxiliary linear head to this representation and also passes it to the final
fusion head.

## `ReferenceEvidenceEncoder`

This encoder processes explicit, low-dimensional measurements relative to a
candidate monitored family and the background reference population.

Default transformation:

```text
sequence evidence [B, 5]             structure evidence [B, 5]
          |                                     |
     Linear(5, 16)                         Linear(5, 16)
          |                                     |
        GELU                                  GELU
          |                                     |
    LayerNorm(16)                        LayerNorm(16)
          |                                     |
     Dropout(0.2)                         Dropout(0.2)
          |                                     |
          +-----------+-------------+-----------+
                      |             |
                    [B, 16]    context [B, 5]
                      |             |
                      +------ concatenate
                                 |
                              [B, 37]
                                 |
                          Linear(37, 32)
                                 |
                               GELU
                                 |
                          LayerNorm(32)
                                 |
                           Dropout(0.2)
                                 |
                   reference representation [B, 32]
```

The context features are not projected before concatenation. Their meanings are
already compact and explicit, and preserving them gives the fusion layer direct
access to quality and availability signals. If later experiments show that
their scale or interactions are problematic, a context projector can be added
as an ablation.

## `Aegis`

`Aegis` owns both encoders and three prediction heads.

### Auxiliary heads

```python
self.embedding_head = nn.Linear(embedding_output_dim, 1)
self.reference_head = nn.Linear(reference_output_dim, 1)
```

These heads produce pathway-specific logits. They make it possible to measure
whether either pathway is useful independently and to apply auxiliary losses
during training.

### Final head

The default representations are concatenated:

```text
embedding representation [B, 128]
reference representation [B, 32]
                 |
             concatenate
                 |
              [B, 160]
                 |
          Linear(160, 64)
                 |
               GELU
                 |
          LayerNorm(64)
                 |
           Dropout(0.2)
                 |
           Linear(64, 1)
                 |
        overall logit [B]
```

The final head can learn interactions such as increasing review evidence when
structural reference similarity is elevated despite weak sequence-derived
evidence. It can also learn to discount structural evidence when context
features indicate low confidence or poor coordinate coverage.

This behavior is possible, not guaranteed. It must be demonstrated through
held-out evaluation and model-ablation results.

## Parameter count

With default dimensions, the model has 152,611 trainable parameters:

| Component | Parameters |
| --- | ---: |
| Embedding encoder | 140,416 |
| Reference encoder | 1,536 |
| Final head | 10,497 |
| Two auxiliary heads | 162 |
| **Total** | **152,611** |

Most capacity lies in the high-dimensional embedding pathway. This imbalance is
one reason the auxiliary reference loss and reference-only ablation are
important: without them, the final model could learn to ignore the smaller
reference pathway.

## Forward-pass example

```python
import torch

from aegisfold.models import Aegis

batch_size = 4
model = Aegis()

output = model(
    sequence_embedding=torch.randn(batch_size, 320),
    structure_embedding=torch.randn(batch_size, 512),
    sequence_evidence=torch.randn(batch_size, 5),
    structure_evidence=torch.randn(batch_size, 5),
    context=torch.randn(batch_size, 5),
)

print(output.overall_logit.shape)      # torch.Size([4])
print(output.embedding_logit.shape)    # torch.Size([4])
print(output.reference_logit.shape)    # torch.Size([4])
```

The random tensors above only verify software behavior. Real evidence features
must have fixed definitions and training-derived normalization.

## Training objective

The model is intended to use the overall loss plus auxiliary pathway losses:

```python
criterion = torch.nn.BCEWithLogitsLoss()

overall_loss = criterion(output.overall_logit, labels.float())
embedding_loss = criterion(output.embedding_logit, labels.float())
reference_loss = criterion(output.reference_logit, labels.float())

loss = overall_loss + 0.25 * embedding_loss + 0.25 * reference_loss
```

In mathematical form:

```text
L = L_overall + lambda_embedding * L_embedding
              + lambda_reference * L_reference
```

The `0.25` weights are initial hyperparameters, not validated biological
constants. They must be tuned using training and validation data without
consulting the final test set.

If only `overall_loss` is used, the auxiliary head parameters are disconnected
from that loss and will not receive gradients. Both encoders and the final head
will still train, but the pathway-specific logits will remain unsuitable for
interpretation. The auxiliary losses are therefore part of the intended
training contract.

## Training and evaluation modes

Training mode enables dropout:

```python
model.train()
```

Evaluation mode disables dropout:

```python
model.eval()
with torch.inference_mode():
    output = model(...)
```

`torch.inference_mode()` also disables gradient bookkeeping, reducing memory
usage. Validation, test evaluation, calibration, and application inference must
all use evaluation mode.

## Calibration and decision policy

The model does not contain probability calibration or operational thresholds.
The intended later flow is:

```text
overall logit
      |
      v
calibrator fitted on validation data
      |
      v
calibrated score
      |
      +-- above review threshold ------------> REVIEW
      +-- reliable and below threshold ------> PASS
      +-- OOD or unreliable structure -------> INCONCLUSIVE
```

The review threshold should be selected for a predefined false-positive rate.
The primary scientific comparison remains recall at that fixed false-positive
rate on sequence-divergent examples.

## Tests

Run focused tests:

```bash
python -m pytest tests/test_fusion_model.py
```

Run the complete repository suite:

```bash
python -m pytest
```

### Encoder representation tests

- `test_embedding_encoder_returns_configured_representation` verifies that the
  embedding encoder returns its configured output width.
- `test_reference_encoder_returns_configured_representation` verifies the same
  contract for explicit evidence.

These tests allow each pathway to be reused and evaluated independently.

### Complete output test

`test_aegis_returns_three_logits_per_protein` confirms that `Aegis` returns an
`AegisOutput` with one overall, embedding, and reference logit per example.

### Configuration test

`test_aegis_supports_configurable_dimensions` builds a deliberately small model
with non-default dimensions. It detects dimensions accidentally hard-coded into
the forward pass.

### Shape-validation tests

Parameterized tests exercise all five inputs:

- `test_aegis_rejects_unbatched_inputs` rejects one-dimensional features.
- `test_aegis_rejects_incorrect_feature_widths` rejects wrong column counts.
- `test_aegis_rejects_mismatched_pathway_batch_sizes` rejects embedding and
  reference batches describing different numbers of examples.

### Gradient-flow test

`test_combined_auxiliary_loss_reaches_every_trainable_parameter` applies the
intended combined loss, runs backpropagation, and confirms that every trainable
parameter receives a gradient. This detects disconnected branches or heads.

It proves that optimization can reach the entire architecture. It does not
prove that the gradients lead to a scientifically useful model.

### Probability-range test

`test_output_probabilities_are_between_zero_and_one` verifies the three sigmoid
convenience properties. It does not establish calibration.

### Evaluation determinism test

`test_evaluation_mode_is_deterministic` uses a high dropout rate, switches the
model to evaluation mode, and verifies that repeated forward passes are equal.

## Required scientific ablations

The complete model must be compared against:

```text
Embedding pathway only
Reference pathway only
Sequence evidence only
Structure evidence only
Complete Aegis model
```

Additional controls should compare classical sequence search, sequence
embedding retrieval, structural search, and structure embedding retrieval.

If complete Aegis outperforms both pathways at the same false-positive rate,
that supports the claim that learned representations and explicit reference
evidence are complementary. If it matches one pathway, the other may add cost
without meaningful value.

## What the code does not implement

`fusion.py` intentionally does not implement:

- reference-library construction;
- nearest-neighbor retrieval;
- sequence or structure alignment;
- evidence-feature calculation;
- feature normalization;
- homology-aware splitting;
- label curation;
- model training;
- probability calibration;
- PASS/REVIEW/INCONCLUSIVE thresholds;
- biological explanations;
- claims of toxicity.

Those pieces determine whether the model is scientifically meaningful. The
current code supplies a tested neural architecture into which validated
embedding and reference evidence can eventually flow.
