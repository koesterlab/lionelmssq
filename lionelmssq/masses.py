import importlib.resources
import polars as pl
from itertools import product

_COLS = ["nucleoside", "monoisotopic_mass", "modification_rate"]


# TODO: Currently, the list of unmodified bases is only defined for RNA;
#  make it universally applicable
UNMODIFIED_BASES = ["A", "C", "G", "U"]


REDUCE_TABLE = True
REDUCE_SET = False
COMPRESSION_RATE = 32

ROUND_DECIMAL = 5  # The precision (after decimal points) to which to consider the nucleoside masses.
# In the nucleoside table, it can happen that the same masses may be reported with different precision values. So UNIQUE MASSES after rounding may not be unique without doing the above step!

TOLERANCE = 1e-3  # For perfect matching, the TOLERANCE should be the
# precision (digits after decimal) to which the masses of nucleosides and sequences are reported, i.e. 1e-(ROUND_DECIMAL)


# Build dict with elemental masses
ELEMENTAL_MASSES = pl.read_csv(
    importlib.resources.files(__package__) / "assets" / "element_masses.tsv",
    separator="\t",
)
ELEMENT_MASSES = {
    row[ELEMENTAL_MASSES.get_column_index("symbol")]: row[
        ELEMENTAL_MASSES.get_column_index("mass")
    ]
    for row in ELEMENTAL_MASSES.iter_rows()
}

# PHOSPHATE_LINK_MASS = 61.95577  # P(30.97389) + 2*O(2*15.99491) - H(1.00783)
PHOSPHATE_LINK_MASS = (
    ELEMENT_MASSES["P"] + 2 * ELEMENT_MASSES["O"] - ELEMENT_MASSES["H+"]
)


# This dictates a relative matching threshold such that we consider abs(sum(masses)/target_mass - 1) < MATCHING_THRESHOLD to be matched!
MATCHING_THRESHOLD = 12e-6
# We choose 20 ppm as the default error from the MS.
# The error is on the higher side than would be for a good calibrated machine (6ppm),
# but in the absence of an experimental measurement of this error, this (very) conservative value works well!


def initialize_nucleotide_df(reduce_set):
    masses = pl.read_csv(
        (
            importlib.resources.files(__package__)
            / "assets"
            / f"{'masses_bases' if reduce_set else 'masses'}.tsv"
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

    return masses, unique_masses, explanation_masses


MASSES, UNIQUE_MASSES, EXPLANATION_MASSES = initialize_nucleotide_df(REDUCE_SET)


# METHOD: Precompute all weight changes caused by breakages and adapt the
# target masses accordingly while finding compositions explaining it.
# We consider tags at the 5'- or 3'-end to be possible breakage options.


def build_breakage_dict(mass_5_prime, mass_3_prime):
    element_masses = ELEMENT_MASSES

    # Initialize dict with masses for 5'-end of fragments
    start_dict = {
        # Remove O from SU and add START tag (without H)
        "START": mass_5_prime - element_masses["O"] - element_masses["H+"],
        # Add H to SU to achieve neutral charge
        "c/y": element_masses["H+"],
    }

    # Initialize dict with masses for 3'-end of fragments
    end_dict = {
        # Remove PO3H from SU and add END tag (without H)
        "END": mass_3_prime
        - element_masses["P"]
        - 3 * element_masses["O"]
        - 2 * element_masses["H+"],
        # Remove H from SU to achieve neutral charge
        "c/y": -element_masses["H+"],
    }

    breakage_dict = {}
    for start, end in list(product(start_dict.keys(), end_dict.keys())):
        val = int((start_dict[start] + end_dict[end]) / TOLERANCE)
        if val not in breakage_dict:
            breakage_dict[val] = []
        breakage_dict[val] += [f"{start}_{end}"]

    return breakage_dict
