import polars as pl
import numpy as np

from spectrseqtools.composition_inference import is_valid_mass
from spectrseqtools.traceback_matrix import CompositionInferrer


MAX_VARIANCE = 1


# METHOD: For each fragmentation option that yields a valid mass (i.e. one that
# has any valid composition) for a given fragment, duplicate the fragment
# and determine its fragmentation-independent standard-unit mass by
# subtracting the weight imposed by the fragmentation.


def classify_fragments(
    fragment_masses,
    inferrer: CompositionInferrer,
    fragmentation_dict: dict,
    output_file_path=None,
    intensity_cutoff=0.5e6,
    mass_cutoff=50000,
) -> pl.DataFrame:
    # If no intensity is given, set it so that all fragments pass the filter
    if "intensity" not in fragment_masses.columns:
        fragment_masses = fragment_masses.with_columns(
            pl.lit(intensity_cutoff * 1.1).alias("intensity"),
        )

    # Rename 'neutral_mass' values from deisotoping to 'observed_mass'
    if "neutral_mass" in fragment_masses.columns:
        fragment_masses = fragment_masses.rename({"neutral_mass": "observed_mass"})

    # Index fragments
    fragment_masses = fragment_masses.with_row_index("fragment_index")

    # Copy each fragment for each unique fragmentation weights and set standard-unit mass
    fragments = pl.concat(
        [
            fragment_masses.with_columns(
                (pl.col("observed_mass") - (weight * inferrer.precision)).alias(
                    "standard_unit_mass"
                ),
                pl.lit(fragmentation[0]).alias("fragmentation"),
            )
            for (weight, fragmentation) in fragmentation_dict.items()
        ]
    )

    # Filter out all fragments with no valid composition
    fragments = (
        fragments.with_columns(
            pl.struct("observed_mass", "standard_unit_mass")
            .map_elements(
                lambda x: is_valid_mass(
                    mass=x["standard_unit_mass"],
                    inferrer=inferrer,
                    threshold=inferrer.tolerance * x["observed_mass"],
                ),
                return_dtype=bool,
            )
            .alias("is_valid")
        )
        .filter(pl.col("is_valid"))
        .drop("is_valid")
    )

    # Determine all fragments that may be singletons
    fragments = fragments.with_columns(
        pl.struct("observed_mass", "standard_unit_mass")
        .map_elements(
            lambda x: is_singleton(
                mass=x["standard_unit_mass"],
                integer_masses=[mass.mass for mass in inferrer.alphabet],
                inferrer=inferrer,
                threshold=inferrer.tolerance * x["observed_mass"],
            ),
            return_dtype=bool,
        )
        .alias("is_singleton")
    )

    # Filter out fragments that have a too high mass or too low intensity
    fragments = (
        fragments.sort(pl.col("standard_unit_mass"))
        .filter(pl.col("intensity") > intensity_cutoff)
        .filter(pl.col("observed_mass") < mass_cutoff)
    )

    # Select highest valid SU mass, i.e. the sequence mass
    mass_cutoff = inferrer.seq.su_mass

    # Filter fragments based on mass cutoff
    fragments = filter_by_intact_mass(mass_cutoff, fragments)

    # Write terminal fragments to file if file name is given
    if output_file_path is not None:
        fragments.write_csv(output_file_path, separator="\t")

    return fragments


def is_singleton(mass, integer_masses, inferrer, threshold=None):
    # Convert the target to an integer for easy operations
    target = int(round(mass / inferrer.precision, 0))

    # Set relative threshold if not given
    if threshold is None:
        threshold = inferrer.tolerance * mass

    # Convert the threshold to integer
    threshold = int(np.ceil(threshold / inferrer.precision))

    # Check whether a singleton mass could be found
    for value in range(target - threshold, target + threshold + 1):
        if value in integer_masses:
            return True
    return False


def filter_by_intact_mass(
    mass_cutoff: float, fragments: pl.DataFrame
) -> pl.DataFrame:
    # Filter out fragments that have a too high SU mass (within variance)
    fragments = fragments.filter(
        pl.col("standard_unit_mass") < mass_cutoff + MAX_VARIANCE
    )

    # Filter out all intact fragments with a too low SU mass (within variance)
    fragments = fragments.filter(
        (pl.col("standard_unit_mass") > mass_cutoff - MAX_VARIANCE)
        | ~(
            pl.col("fragmentation").str.contains("START")
            & pl.col("fragmentation").str.contains("END")
        )
    )

    return fragments
