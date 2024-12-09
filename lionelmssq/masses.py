import importlib.resources
import polars as pl

_COLS = ["nucleoside", "monoisotopic_mass"]

MASSES = pl.read_csv(
    (importlib.resources.files(__package__) / "assets" / "masses.tsv"),
    separator="\t",
    # ("assets/masses_all.tsv"), separator="\t"
)
# TODO: Change the file to masses_all.tsv for all modifications. Note: "masses_all.tsv" has multiples nucleosides with the same mass!

assert MASSES.columns == _COLS

# TODO: Add the appropriate backbone masses and the terminal extra masses to the nucleosides!

UNIQUE_MASSES = (
    MASSES.group_by(
        "monoisotopic_mass", maintain_order=True
    )  # TODO: But this groups by the float values of the masses without any precision! Change this!
    .first()  # TODO: Change this to "agg" instead of "first", so that we can get all the nucleosides with the same mass!
    # .agg(pl.col("nucleoside"))
    .select(pl.col(_COLS))
)

# print(EXPLANATION_MASSES.select(pl.col("nucleoside")).dtypes[0])

TOLERANCE = 1e-3

EXPLANATION_MASSES = UNIQUE_MASSES.with_columns(
    (pl.col("monoisotopic_mass") / TOLERANCE).round(0).cast(pl.Int64).alias("coins")
)
