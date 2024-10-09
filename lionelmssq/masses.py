import importlib.resources
import polars as pl

_COLS = ["nucleoside", "monoisotopic_mass"]

MASSES = pl.read_csv((importlib.resources.files(__package__) / "assets" / "masses.tsv"), separator="\t")

assert MASSES.columns == _COLS


UNIQUE_MASSES = MASSES.group_by("monoisotopic_mass", maintain_order=True).first().select(pl.col(_COLS))