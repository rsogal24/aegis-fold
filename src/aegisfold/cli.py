"""Command-line interface for AegisFold research workflows."""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from aegisfold import __version__
from aegisfold.io import save_embedding

app = typer.Typer(no_args_is_help=True, help="AegisFold research utilities")
console = Console()


@app.command()
def info() -> None:
    """Show the installed AegisFold version."""

    console.print(f"AegisFold {__version__}")


@app.command("embed-sequence")
def embed_sequence(
    sequence: Annotated[str, typer.Argument(help="Protein amino-acid sequence")],
    output: Annotated[Path, typer.Option("--output", "-o")] = Path(
        "artifacts/embeddings/sequence.pt"
    ),
    model_name: Annotated[str, typer.Option("--model")] = "facebook/esm2_t6_8M_UR50D",
    device: Annotated[str | None, typer.Option("--device")] = None,
) -> None:
    """Generate one pooled ESM-2 sequence embedding."""

    from aegisfold.embeddings.sequence import ESM2Embedder

    result = ESM2Embedder(model_name=model_name, device=device).embed(sequence)
    path = save_embedding(result, output)
    console.print(
        f"Saved {result.dimension}-dimensional sequence embedding for "
        f"{result.residue_count} residues to {path}"
    )


@app.command("embed-structure")
def embed_structure(
    pdb_path: Annotated[Path, typer.Argument(help="Input PDB file")],
    chain: Annotated[str, typer.Option("--chain", "-c")] = "A",
    output: Annotated[Path, typer.Option("--output", "-o")] = Path(
        "artifacts/embeddings/structure.pt"
    ),
    device: Annotated[str, typer.Option("--device")] = "cpu",
) -> None:
    """Generate one pooled ESM-IF1 structure embedding."""

    from aegisfold.embeddings.structure import ESMIF1Embedder

    result = ESMIF1Embedder(device=device).embed_pdb(pdb_path, chain)
    path = save_embedding(result, output)
    console.print(
        f"Saved {result.dimension}-dimensional structure embedding for "
        f"{result.residue_count} residues to {path}"
    )


if __name__ == "__main__":
    app()

