# AegisFold

AegisFold is a defensive research project evaluating whether structural evidence
improves screening of sequence-divergent proteins over sequence-only methods.
The public benchmark is designed around benign proxy families and reports
calibrated review signals rather than categorical toxicity claims.

## Research plan

1. Establish sequence-similarity and pretrained sequence-embedding baselines.
2. Add structural-search and pretrained structural-embedding baselines.
3. Train and evaluate sequence-structure fusion under homology-aware splits.
4. Deploy a screening application only if fusion provides a meaningful benefit.

## Local setup

AegisFold uses Python 3.11 because ESM-IF1 depends on a legacy scientific stack.

```bash
/opt/homebrew/bin/python3.11 -m venv aegis_env
source aegis_env/bin/activate
python -m pip install --upgrade pip
python -m pip install --no-cache-dir -r requirements.txt
python -m pip install --no-deps -e .
```

Confirm the package is available:

```bash
aegisfold info
```

## Encoder smoke tests

Generate a pooled ESM-2 representation from a short example sequence:

```bash
python scripts/verify_sequence_embedding.py
```

Generate a pooled ESM-IF1 representation from chain A of a local PDB file:

```bash
python scripts/verify_structure_embedding.py path/to/protein.pdb --chain A
```

The equivalent CLI commands can persist embeddings under the ignored
`artifacts/` directory:

```bash
aegisfold embed-sequence "MKTIIALSYIFCLVFADYKDDDDK"
aegisfold embed-structure path/to/protein.pdb --chain A
```

## Repository boundaries

Virtual environments, model weights, raw/processed datasets, checkpoints, and
generated artifacts are intentionally excluded from Git. Model weights are
loaded from the local Hugging Face or PyTorch cache at runtime.

## Developer documentation

- [Fusion model and test guide](docs/fusion-model.md)
