import importlib.resources
import polars as pl
import os
from itertools import product

_COLS = ["nucleoside", "monoisotopic_mass"]

REDUCE_TABLE = True
REDUCE_SET = False
COMPRESSION_RATE = 32

MASSES = pl.read_csv(
    (
        importlib.resources.files(__package__)
        / "assets"
        / f"{"masses_bases" if REDUCE_SET else "masses"}.tsv"
    ),
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

TOLERANCE = 1e-3  # For perfect matching, the TOLERANCE should be the
# precision (digits after decimal) to which the masses of nucleosides and sequences are reported, i.e. 1e-(ROUND_DECIMAL)

TABLE_PATH = (
    f"dp_table/{'reduced' if REDUCE_TABLE else 'full'}_table."
    f"{'reduced' if REDUCE_SET else 'full'}_set/"
    f"tol_{TOLERANCE:.0E}"
)

if not os.path.exists("dp_table"):
    os.makedirs("dp_table")

subdir_path = (
    f"dp_table/{'reduced' if REDUCE_TABLE else 'full'}_table."
    f"{'reduced' if REDUCE_SET else 'full'}_set"
)

if not os.path.exists(subdir_path):
    os.makedirs(subdir_path)

print(TABLE_PATH)

PHOSPHATE_LINK_MASS = 61.95577  # P(30.97389) + 2*O(2*15.99491) + H(1.00783)

# Additional weights for different breakage options
START_OPTIONS = pl.read_csv(
    importlib.resources.files(__package__)/"assets"/"5_prime_end_breakage.tsv",
    separator="\t",
)
END_OPTIONS = pl.read_csv(
    importlib.resources.files(__package__)/"assets"/"3_prime_end_breakage.tsv",
    separator="\t",
)

BREAKAGES = {}
for start, end in list(product(
        START_OPTIONS.select("name").to_series().to_list(),
        END_OPTIONS.select("name").to_series().to_list())):
    val = (
        START_OPTIONS.filter(pl.col("name")==start).select("weight").item() +
        END_OPTIONS.filter(pl.col("name")==end).select("weight").item())
    if val not in BREAKAGES:
        BREAKAGES[val] = []
    BREAKAGES[val] += [f"{start}_{end}"]

# BREAKAGES = {0: ["c/y_c/y"]}
BREAKAGES = {int(val / TOLERANCE): BREAKAGES[val] for val in BREAKAGES.keys()}

# This dictates a relative matching threshold such that we consider abs(sum(masses)/target_mass - 1) < MATCHING_THRESHOLD to be matched!
MATCHING_THRESHOLD = 20e-6
# We choose 20 ppm as the default error from the MS.
# The error is on the higher side than would be for a good calibrated machine (6ppm),
# but in the absence of an experimental measurement of this error, this (very) conservative value works well!

EXPLANATION_MASSES = UNIQUE_MASSES.with_columns(
    ((pl.col("monoisotopic_mass") + PHOSPHATE_LINK_MASS) / TOLERANCE)
    .round(0)
    .cast(pl.Int64)
    .alias("tolerated_integer_masses")
)

MAX_MASS = EXPLANATION_MASSES.select(
    pl.col("tolerated_integer_masses")
).max().item() * (12 if REDUCE_TABLE else 35)
