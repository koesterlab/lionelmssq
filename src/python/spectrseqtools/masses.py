import importlib.resources
import polars as pl
from itertools import product

_COLS = [
    "id",
    "canonical_name",
    "monoisotopic_mass",
    "modification_rate",
    "encoding",
]


# TODO: Currently, the list of unmodified bases is only defined for RNA;
#  make it universally applicable
UNMODIFIED_BASES = ["A", "C", "G", "U"]

# Set default value for intensity cutoff
DEFAULT_INTENSITY_CUTOFF = 115000

# Set fragmentation dict modus (full vs only c/y)
FULL_FRAGMENTATION_DICT = False


# Set number of binary-compressed masses per integer cell in traceback matrix
COMPRESSION_RATE = 32


# Set the number of decimal places up to which to consider nucleoside masses
DECIMAL_PLACES = 3

# Set precision for calculations
PRECISION = 10 ** (-DECIMAL_PLACES)

# Set relative tolerance such that we consider
# abs(sum(masses)/target_mass - 1) < TOLERANCE for matching
# Note that the error is on the higher side than would be for a good
# calibrated machine (10 ppm), but in the absence of an experimental
# measurement of this error, this conservative value works well
TOLERANCE = 10e-6


# Build dict with elemental masses
elements = pl.read_csv(
    importlib.resources.files(__package__) / "assets" / "element_masses.tsv",
    separator="\t",
)
ELEMENT_MASSES = {
    row[elements.get_column_index("symbol")]: row[elements.get_column_index("mass")]
    for row in elements.iter_rows()
}


# METHOD: Precompute all weight changes caused by fragmentation and adapt the
# target masses accordingly while finding compositions explaining it.
# We consider tags at the 5'- or 3'-end to be possible fragmentation options.


def build_fragmentation_dict(start_tag, end_tag):
    element_masses = ELEMENT_MASSES

    # Initialize dict with masses for 5'-end of fragments
    start_dict = {
        # Remove O from SU and add START tag (without H)
        "START": start_tag - element_masses["O"] - element_masses["H+"],
        # Add H to SU to achieve neutral charge
        "c/y": element_masses["H+"],
    }

    # Initialize dict with masses for 3'-end of fragments
    end_dict = {
        # Remove PO3H from SU and add END tag (without H)
        "END": end_tag
        - element_masses["P"]
        - 3 * element_masses["O"]
        - 2 * element_masses["H+"],
        # Remove H from SU to achieve neutral charge
        "c/y": -element_masses["H+"],
    }

    # Add a/w-, b/x-, and d/z-fragmentation for full dict version
    if FULL_FRAGMENTATION_DICT:
        # Add PO3H2 to SU to achieve neutral charge
        start_dict["a/w"] = (
            element_masses["P"] + 3 * element_masses["O"] + 2 * element_masses["H+"]
        )
        # Add P2O to SU to achieve neutral charge
        start_dict["b/x"] = element_masses["P"] + 2 * element_masses["O"]
        # Remove OH from SU to achieve neutral charge
        start_dict["d/z"] = -(element_masses["O"] + element_masses["H+"])

        # Remove PO3H2 from SU to achieve neutral charge
        end_dict["a/w"] = -(
            element_masses["P"] + 3 * element_masses["O"] + 2 * element_masses["H+"]
        )
        # Remove P2O from SU to achieve neutral charge
        end_dict["b/x"] = -(element_masses["P"] + 2 * element_masses["O"])
        # Add OH to SU to achieve neutral charge
        end_dict["d/z"] = element_masses["O"] + element_masses["H+"]

    # Collect all unique fragmentation-related mass combinations in dict
    fragmentation_dict = {}
    for start, end in list(product(start_dict.keys(), end_dict.keys())):
        val = int((start_dict[start] + end_dict[end]) / PRECISION)
        if val not in fragmentation_dict:
            fragmentation_dict[val] = []
        fragmentation_dict[val] += [f"{start}_{end}"]

    return fragmentation_dict
