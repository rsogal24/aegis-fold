"""ESM-IF1 structure embedding extraction."""

from pathlib import Path

import numpy as np
import torch

from aegisfold.embeddings.base import EmbeddingResult, masked_mean

DEFAULT_STRUCTURE_MODEL = "esm_if1_gvp4_t16_142M_UR50"


class ESMIF1Embedder:
    """Lazy-loading ESM-IF1 encoder for protein backbone coordinates."""

    def __init__(self, device: str = "cpu"):
        self.model_name = DEFAULT_STRUCTURE_MODEL
        self.device = torch.device(device)
        self._model = None
        self._alphabet = None

    def load(self) -> None:
        """Load the cached ESM-IF1 checkpoint."""

        if self._model is not None:
            return
        import esm

        model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
        self._model = model.eval().to(self.device)
        self._alphabet = alphabet

    def embed_pdb(self, pdb_path: str | Path, chain_id: str) -> EmbeddingResult:
        """Extract and pool ESM-IF1 representations for one PDB chain."""

        path = Path(pdb_path)
        if not path.is_file():
            raise FileNotFoundError(f"PDB file not found: {path}")
        if not chain_id:
            raise ValueError("chain_id cannot be empty")

        self.load()
        assert self._model is not None
        assert self._alphabet is not None

        import esm.inverse_folding

        structure = esm.inverse_folding.util.load_structure(str(path), chain_id)
        coords, sequence = esm.inverse_folding.util.extract_coords_from_structure(structure)
        residue_embeddings = esm.inverse_folding.util.get_encoder_output(
            self._model,
            self._alphabet,
            coords,
        )
        coordinate_mask = np.isfinite(coords).all(axis=(1, 2))
        mask = torch.as_tensor(coordinate_mask, device=residue_embeddings.device)
        vector = masked_mean(residue_embeddings, mask).detach().cpu()
        return EmbeddingResult(
            vector=vector,
            model_name=self.model_name,
            residue_count=len(sequence),
        )

