import importlib.resources
import polars as pl
import os

_COLS = ["nucleoside", "monoisotopic_mass"]

REDUCE_TABLE = True
REDUCE_SET = False

MASSES = pl.read_csv(
    (importlib.resources.files(__package__) / "assets" /
     f"{"masses_bases" if REDUCE_SET else "masses"}.tsv"),
    separator="\t",
)
# Note: "masses.tsv" has multiples nucleosides with the same mass!

assert MASSES.columns == _COLS

# Uncomment if we only want to consider the natural (unmodified) nucleosides i.e. [A,G,U,C]!
# MASSES = MASSES.filter(pl.col("nucleoside").is_in(["A", "G", "U", "C"]))

ROUND_DECIMAL = 5  # The precision (after decimal points) to which to consider the nucleoside masses.
# In the nucleoside table, it can happen that the same masses may be reported with different precision values. So UNIQUE MASSES after rounding may not be unique without doing the above step!

MASSES = MASSES.with_columns(pl.col("monoisotopic_mass").round(ROUND_DECIMAL))

# TODO: Add the appropriate backbone masses and the terminal extra masses to the nucleosides!
UNIQUE_MASSES = (
    MASSES.group_by("monoisotopic_mass", maintain_order=True)
    .first()
    .select(pl.col(_COLS))
)

# For mass explanation for ladder building, to convert the masses to an integer value for the DP algorithm!

TOLERANCE = 1e-5  # For perfect matching, the TOLERANCE should be the precision (digits after decimal) to which the masses of nucleosides and sequences are reported, i.e. 1e-(ROUND_DECIMAL)

TABLE_PATH = (f"dp_table/{'reduced' if REDUCE_TABLE else 'full'}_table."
              f"{'reduced' if REDUCE_SET else 'full'}_set/"
              f"tolerance_{TOLERANCE}")

if not os.path.exists("dp_table"):
    os.makedirs("dp_table")

subdir_path = (
    f"dp_table/{'reduced' if REDUCE_TABLE else 'full'}_table."
    f"{'reduced' if REDUCE_SET else 'full'}_set"
)

if not os.path.exists(subdir_path):
    os.makedirs(subdir_path)

print(TABLE_PATH)

# TODO: Use monoisotopic masses to calculate a more accurate link mass
# Mass of P (31) + 2*O (2*16) - H (1)
# PHOSPHATE_LINK_MASS = 0
PHOSPHATE_LINK_MASS = 62

MATCHING_THRESHOLD = 10  # This dictates a matching threshold such that we consider -MATCHING_THRESHOLD < (sum(masses) - target_mass) < MATCHING_THRESHOLD to be matched!
# If TOLERANCE < num_of_decimals in reported masses, then MATCHING_THRESHOLD should at least be greater or equal than the number of nucleotides expected for a target mass!

EXPLANATION_MASSES = UNIQUE_MASSES.with_columns(
    ((pl.col("monoisotopic_mass") + PHOSPHATE_LINK_MASS)  / TOLERANCE)
    .round(0)
    .cast(pl.Int64)
    .alias("tolerated_integer_masses")
)

MAX_MASS = EXPLANATION_MASSES.select(pl.col(
    "tolerated_integer_masses")).max().item() * (12 if REDUCE_TABLE else 35)