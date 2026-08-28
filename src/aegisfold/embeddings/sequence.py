"""ESM-2 sequence embedding extraction."""

import re

import torch

from aegisfold.embeddings.base import EmbeddingResult, masked_mean

DEFAULT_SEQUENCE_MODEL = "facebook/esm2_t6_8M_UR50D"
_VALID_SEQUENCE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWYBXZOU]+$")


def normalize_sequence(sequence: str) -> str:
    """Normalize whitespace and validate an amino-acid sequence."""

    normalized = "".join(sequence.split()).upper()
    if not normalized:
        raise ValueError("sequence cannot be empty")
    if not _VALID_SEQUENCE.fullmatch(normalized):
        raise ValueError("sequence contains unsupported characters")
    return normalized


def default_device() -> torch.device:
    """Choose the best generally available inference device."""

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ESM2Embedder:
    """Lazy-loading ESM-2 encoder with masked mean pooling."""

    def __init__(self, model_name: str = DEFAULT_SEQUENCE_MODEL, device: str | None = None):
        self.model_name = model_name
        self.device = torch.device(device) if device else default_device()
        self._tokenizer = None
        self._model = None

    def load(self) -> None:
        """Load the tokenizer and cached model checkpoint."""

        if self._model is not None:
            return
        from transformers import AutoModel, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(
            self.model_name,
            add_pooling_layer=False,
        ).to(self.device)
        self._model.eval()

    def embed(self, sequence: str) -> EmbeddingResult:
        """Convert a protein sequence into one fixed-size vector."""

        normalized = normalize_sequence(sequence)
        self.load()
        assert self._tokenizer is not None
        assert self._model is not None

        encoded = self._tokenizer(
            normalized,
            return_tensors="pt",
            return_special_tokens_mask=True,
        )
        special_tokens = encoded.pop("special_tokens_mask").to(self.device)
        model_inputs = {name: value.to(self.device) for name, value in encoded.items()}

        with torch.inference_mode():
            hidden = self._model(**model_inputs).last_hidden_state[0]

        valid_mask = model_inputs["attention_mask"][0].bool() & ~special_tokens[0].bool()
        vector = masked_mean(hidden, valid_mask).detach().cpu()
        return EmbeddingResult(
            vector=vector,
            model_name=self.model_name,
            residue_count=len(normalized),
        )

