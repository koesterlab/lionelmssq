import importlib.resources
import polars as pl
import os
from itertools import product

_COLS = ["nucleoside", "monoisotopic_mass"]

REDUCE_TABLE = True
REDUCE_SET = False
USE_BITS = True

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

TOLERANCE = 1e-5  # For perfect matching, the TOLERANCE should be the precision (digits after decimal) to which the masses of nucleosides and sequences are reported, i.e. 1e-(ROUND_DECIMAL)

TABLE_PATH = (
    f"dp_table/{'reduced' if REDUCE_TABLE else 'full'}_table."
    f"{'reduced' if REDUCE_SET else 'full'}_set/"
    f"tolerance_{TOLERANCE}"
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
START_OPTIONS = {
    "START": 537.11887,  # current START tag
    "a/w": 79.96633,  # additional P+3O+H
    "b/x": 63.97142,  # additional P+2O+H
    "c/y": 0,  # neutral state (for standard unit)
    "d/z": -15.99491,  # lost O
}

END_OPTIONS = {
    "END": 373.16714,  # current END tag
    "d/z": 15.99491,  # additional O
    "c/y": 0,  # neutral state (for standard unit)
    "c/y-cyclization": -18.01056,  # lost O+2H
    "b/x": -63.97142,  # lost P+2O+H
    "a/w": -79.96633,  # lost P+3O+H
}

BREAKAGES = {}
for start, end in list(product(START_OPTIONS, END_OPTIONS)):
    val = START_OPTIONS[start] + END_OPTIONS[end]
    if val not in BREAKAGES:
        BREAKAGES[val] = []
    BREAKAGES[val] += [f"{start}_{end}"]

# BREAKAGES = {0: ["c/y_c/y"]}
BREAKAGES = {int(val / TOLERANCE): BREAKAGES[val] for val in BREAKAGES.keys()}

MATCHING_THRESHOLD = 10  # This dictates a matching threshold such that we consider -MATCHING_THRESHOLD < (sum(masses) - target_mass) < MATCHING_THRESHOLD to be matched!
# If TOLERANCE < num_of_decimals in reported masses, then MATCHING_THRESHOLD should at least be greater or equal than the number of nucleotides expected for a target mass!

EXPLANATION_MASSES = UNIQUE_MASSES.with_columns(
    ((pl.col("monoisotopic_mass") + PHOSPHATE_LINK_MASS) / TOLERANCE)
    .round(0)
    .cast(pl.Int64)
    .alias("tolerated_integer_masses")
)

MAX_MASS = EXPLANATION_MASSES.select(
    pl.col("tolerated_integer_masses")
).max().item() * (12 if REDUCE_TABLE else 35)
