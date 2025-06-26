import importlib.resources
import polars as pl
import os
from itertools import product
from platformdirs import user_cache_dir

_COLS = ["nucleoside", "monoisotopic_mass"]

REDUCE_TABLE = True
REDUCE_SET = False
COMPRESSION_RATE = 32

ROUND_DECIMAL = 5  # The precision (after decimal points) to which to consider the nucleoside masses.
# In the nucleoside table, it can happen that the same masses may be reported with different precision values. So UNIQUE MASSES after rounding may not be unique without doing the above step!

TOLERANCE = 1e-3  # For perfect matching, the TOLERANCE should be the
# precision (digits after decimal) to which the masses of nucleosides and sequences are reported, i.e. 1e-(ROUND_DECIMAL)


# Set OS-independent cache directory for DP tables
TABLE_DIR = user_cache_dir(
    appname="lionelmssq/dp_table", version="1.0", ensure_exists=True
)

# Set path for DP table
TABLE_PATH = (
    f"{TABLE_DIR}/{'reduced' if REDUCE_TABLE else 'full'}_table."
    f"{'reduced' if REDUCE_SET else 'full'}_set/"
    f"tol_{TOLERANCE:.0E}"
)

# Create directory for DP table if it does not already exist
subdir = "/".join(TABLE_PATH.split("/")[:-1])
if not os.path.exists(subdir):
    os.makedirs(subdir)

print(TABLE_PATH)


PHOSPHATE_LINK_MASS = 61.95577  # P(30.97389) + 2*O(2*15.99491) + H(1.00783)


# This dictates a relative matching threshold such that we consider abs(sum(masses)/target_mass - 1) < MATCHING_THRESHOLD to be matched!
MATCHING_THRESHOLD = 20e-6
# We choose 20 ppm as the default error from the MS.
# The error is on the higher side than would be for a good calibrated machine (6ppm),
# but in the absence of an experimental measurement of this error, this (very) conservative value works well!


# METHOD: Precompute all weight changes caused by breakages and adapt the
# target masses accordingly while finding compositions explaining it.
# We consider tags at the 5'- or 3'-end to be possible breakage options.

# Load additional weights for different breakage options
START_OPTIONS = pl.read_csv(
    importlib.resources.files(__package__)
    / "assets"
    / "5_prime_end_breakage.experimental.tsv",
    separator="\t",
)
END_OPTIONS = pl.read_csv(
    importlib.resources.files(__package__)
    / "assets"
    / "3_prime_end_breakage.experimental.tsv",
    separator="\t",
)

# Compute dict assigning each possible breakage-induced weight change its list
# of associated breakage pairs (i.e. 5'- and 3'-end) that can result into it
BREAKAGES = {}
for start, end in list(
    product(
        START_OPTIONS.select("name").to_series().to_list(),
        END_OPTIONS.select("name").to_series().to_list(),
    )
):
    val = (
        START_OPTIONS.filter(pl.col("name") == start).select("weight").item()
        + END_OPTIONS.filter(pl.col("name") == end).select("weight").item()
    )
    if val not in BREAKAGES:
        BREAKAGES[val] = []
    BREAKAGES[val] += [f"{start}_{end}"]

BREAKAGES = {int(val / TOLERANCE): BREAKAGES[val] for val in BREAKAGES.keys()}


def initialize_nucleotide_df(reduce_table, reduce_set):
    masses = pl.read_csv(
        (
            importlib.resources.files(__package__)
            / "assets"
            / f"{"masses_bases" if reduce_set else "masses"}.tsv"
        ),
        separator="\t",
    )
    # Note: "masses.tsv" has multiples nucleosides with the same mass!

    assert masses.columns == _COLS

    masses = masses.with_columns(pl.col("monoisotopic_mass").round(ROUND_DECIMAL))

    unique_masses = (
        masses.group_by("monoisotopic_mass", maintain_order=True)
        .first()
        .select(pl.col(_COLS))
    )
    # For mass explanation for ladder building, to convert the masses to an integer value for the DP algorithm!

    explanation_masses = unique_masses.with_columns(
        ((pl.col("monoisotopic_mass") + PHOSPHATE_LINK_MASS) / TOLERANCE)
        .round(0)
        .cast(pl.Int64)
        .alias("tolerated_integer_masses")
    )

    max_mass = explanation_masses.select(
        pl.col("tolerated_integer_masses")
    ).max().item() * (12 if reduce_table else 35)

    return masses, unique_masses, explanation_masses, max_mass


MASSES, UNIQUE_MASSES, EXPLANATION_MASSES, MAX_MASS = initialize_nucleotide_df(
    REDUCE_TABLE, REDUCE_SET
)
