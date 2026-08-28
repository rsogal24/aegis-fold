"""Generate a small ESM-2 embedding as an installation smoke test."""

from aegisfold.embeddings.sequence import ESM2Embedder


def main() -> None:
    sequence = "MKTIIALSYIFCLVFADYKDDDDK"
    result = ESM2Embedder().embed(sequence)
    print(
        f"ESM-2 OK: residues={result.residue_count}, "
        f"dimension={result.dimension}, shape={tuple(result.vector.shape)}"
    )


if __name__ == "__main__":
    main()

