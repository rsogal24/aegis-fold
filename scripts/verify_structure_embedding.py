"""Generate an ESM-IF1 embedding from a local PDB chain."""

import argparse

from aegisfold.embeddings.structure import ESMIF1Embedder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdb_path", help="Path to a local PDB file")
    parser.add_argument("--chain", default="A", help="PDB chain identifier")
    args = parser.parse_args()

    result = ESMIF1Embedder().embed_pdb(args.pdb_path, args.chain)
    print(
        f"ESM-IF1 OK: residues={result.residue_count}, "
        f"dimension={result.dimension}, shape={tuple(result.vector.shape)}"
    )


if __name__ == "__main__":
    main()

