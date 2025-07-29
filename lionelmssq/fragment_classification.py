import polars as pl
import numpy as np

from lionelmssq.masses import (
    BREAKAGES,
)
from lionelmssq.mass_explanation import (
    is_valid_mass,
    is_valid_su_mass,
)
from lionelmssq.mass_table import DynamicProgrammingTable


def counts_subset(explanation, ms1_explanations):
    """
    Check if the counts of the different nucleosides are less than or equal to the respective counts in the MS1 mass explanation
    """
    for ms1_explanation in ms1_explanations:
        if all(
            explanation.count(nucleoside) <= ms1_explanation.count(nucleoside)
            for nucleoside in explanation
        ):
            return True
    return False


# TODO: Decide whether to keep MS1-composition filter (would require
#  prior knowledge about sequence length)
# def filter_by_ms1_composition(
#     fragment_mass, dp_table, ms1_mass, ms1_compositions, threshold
# ):
#     compositions = [
#         entry
#         for entry in explain_mass_with_dp(
#             mass=fragment_mass,
#             with_memo=True,
#             dp_table=dp_table,
#             compression_rate=dp_table.compression_per_cell,
#             threshold=threshold,
#         )
#     ]
#
#     # Remove all full-sequence explanations that differ in mass by more than the threshold
#     if abs(fragment_mass / ms1_mass - 1) > threshold:
#         compositions = {
#             composition
#             for composition in compositions
#             if composition.breakage != "START_END"
#         }
#
#     # Remove all explanations that are not a subset of any MS1 composition
#     for composition in compositions:
#         composition.explanations = {
#             explanation
#             for explanation in composition.explanations
#             if counts_subset(explanation, ms1_compositions.explanations)
#         }
#     return compositions


def is_complete_fragment_candidate(mass, dp_table):
    # Reduce breakage options to only allow complete sequences
    breakages = {
        breakage_weight: [
            breakage
            for breakage in BREAKAGES[breakage_weight]
            if breakage == "START_END"
        ]
        for breakage_weight in BREAKAGES
    }
    # Return flag whether the mass is valid with any remaining breakage options
    return is_valid_mass(mass=mass, dp_table=dp_table, breakages=breakages)


def is_start_fragment_candidate(mass, dp_table):
    # Reduce breakage options to only allow terminal fragments (start only)
    breakages = {
        breakage_weight: [
            breakage
            for breakage in BREAKAGES[breakage_weight]
            if "START_" in breakage and breakage != "START_END"
        ]
        for breakage_weight in BREAKAGES
    }
    # Return flag whether the mass is valid with any remaining breakage options
    return is_valid_mass(mass=mass, dp_table=dp_table, breakages=breakages)


def is_end_fragment_candidate(mass, dp_table):
    # Reduce breakage options to only allow terminal fragments (end only)
    breakages = {
        breakage_weight: [
            breakage
            for breakage in BREAKAGES[breakage_weight]
            if "_END" in breakage and breakage != "START_END"
        ]
        for breakage_weight in BREAKAGES
    }
    # Return flag whether the mass is valid with any remaining breakage options
    return is_valid_mass(mass=mass, dp_table=dp_table, breakages=breakages)


def is_internal_fragment_candidate(mass, dp_table):
    # Reduce breakage options to only allow internal fragments
    breakages = {
        breakage_weight: [
            breakage
            for breakage in BREAKAGES[breakage_weight]
            if not ("START_" in breakage or "_END" in breakage)
        ]
        for breakage_weight in BREAKAGES
    }
    # Return flag whether the mass is valid with any remaining breakage options
    return is_valid_mass(mass=mass, dp_table=dp_table, breakages=breakages)


def is_singleton_candidate(mass, integer_masses, dp_table, threshold=None):
    # Convert the target to an integer for easy operations
    target = int(round(mass / dp_table.precision, 0))

    # Set relative threshold if not given
    if threshold is None:
        threshold = dp_table.tolerance * mass

    # Convert the threshold to integer
    threshold = int(np.ceil(threshold / dp_table.precision))

    # Check for each breakage whether a singleton mass could be found
    for breakage_weight in BREAKAGES:
        for value in range(target - threshold, target + threshold + 1):
            if value - breakage_weight in integer_masses:
                return True
    return False


def mark_terminal_fragment_candidates(
    fragment_masses,
    dp_table: DynamicProgrammingTable,
    ms1_mass=None,
    output_file_path=None,
    mass_column_name="neutral_mass",
    output_mass_column_name="observed_mass",
    intensity_cutoff=0.5e6,
    mass_cutoff=50000,
    # ms1_mass_deviations_allowed=0.01,
):
    neutral_masses = (
        fragment_masses.select(pl.col(mass_column_name)).to_series().to_list()
    )

    integer_masses = [mass.mass for mass in dp_table.masses]

    ms1_compositions = None
    # if ms1_mass:
    #     # Compute all compositions for the full sequence (with both tags)
    #     ms1_compositions = [
    #         entry
    #         for entry in explain_mass_with_dp(
    #             mass=ms1_mass,
    #             with_memo=True,
    #             dp_table=dp_table,
    #             compression_rate=dp_table.compression_per_cell,
    #             threshold=threshold,
    #         )
    #         if entry.breakage == "START_END"
    #     ][0]
    #
    #     if len(ms1_compositions) == 0:
    #         print("The MS1 mass could not be explained by the nucleosides!")

    # Filter compositions by MS1 mass
    # TODO: Using the MS1 filter here is not compatible with only doing a table look-up;
    #  check whether it is still helpful when having composition profiles
    # for mass in neutral_masses:
    #     if ms1_compositions:
    #         compositions = filter_by_ms1_composition(
    #             fragment_mass=mass, dp_table=dp_table, ms1_mass=ms1_mass,
    #             ms1_compositions=ms1_compositions,
    #             threshold=ms1_mass_deviations_allowed
    #         )

    fragments = fragment_masses.with_columns(
        pl.Series(neutral_masses).alias(output_mass_column_name)
    )

    # Determine all fragments that may be terminal ones (start only)
    fragments = fragments.with_columns(
        pl.col(mass_column_name)
        .map_elements(
            lambda x: is_start_fragment_candidate(x, dp_table=dp_table),
            return_dtype=bool,
        )
        .alias("is_start")
    )

    # Determine all fragments that may be terminal ones (end only)
    fragments = fragments.with_columns(
        pl.col(mass_column_name)
        .map_elements(
            lambda x: is_end_fragment_candidate(x, dp_table=dp_table),
            return_dtype=bool,
        )
        .alias("is_end")
    )

    # Determine all fragments that may be singletons (with or without tags)
    fragments = fragments.with_columns(
        pl.col(mass_column_name)
        .map_elements(
            lambda x: is_singleton_candidate(
                x, integer_masses=integer_masses, dp_table=dp_table
            ),
            return_dtype=bool,
        )
        .alias("single_nucleoside")
    )

    # Determine all fragments that may be the complete sequence
    fragments = fragments.with_columns(
        pl.col(mass_column_name)
        .map_elements(
            lambda x: is_complete_fragment_candidate(x, dp_table=dp_table),
            return_dtype=bool,
        )
        .alias("is_start_end")
    )

    # Determine all fragments that may be internal ones
    fragments = fragments.with_columns(
        pl.col(mass_column_name)
        .map_elements(
            lambda x: is_internal_fragment_candidate(x, dp_table=dp_table),
            return_dtype=bool,
        )
        .alias("is_internal")
    )

    # Set all other terminal/internal indicators to False for full-sequence candidates
    # TODO: Figure out why this is needed for skeleton
    fragments = fragments.with_columns(
        pl.col("is_start") & ~pl.col("is_start_end").alias("is_start"),
        pl.col("is_end") & ~pl.col("is_start_end").alias("is_end"),
        pl.col("is_internal") & ~pl.col("is_start_end").alias("is_internal"),
    )

    # Use MS1 mass as an additional cutoff for the fragment masses
    if ms1_compositions:
        mass_cutoff = 1.01 * ms1_mass

    # Filter out fragments that are either non-terminal or have a too high
    # mass or too low intensity
    fragments = (
        fragments.filter(
            (
                pl.col("is_start")
                | pl.col("is_end")
                | pl.col("is_start_end")
                | pl.col("is_internal")
            )
        )
        .sort(pl.col("observed_mass"))
        .filter(pl.col("intensity") > intensity_cutoff)
        .filter(pl.col("observed_mass") < mass_cutoff)
    )

    # Write terminal fragments to file if file name is given
    if output_file_path is not None:
        fragments.write_csv(output_file_path, separator="\t")

    return fragments


def classify_fragments(
    fragment_masses,
    dp_table: DynamicProgrammingTable,
    breakage_dict=BREAKAGES,
    output_file_path=None,
    intensity_cutoff=0.5e6,
    mass_cutoff=50000,
) -> pl.DataFrame:
    if "observed_mass" in fragment_masses.columns:
        fragment_masses = fragment_masses.with_columns(
            pl.lit(intensity_cutoff * 1.1).alias("intensity"),
        )
    else:
        fragment_masses = fragment_masses.with_columns(
            pl.col("neutral_mass").alias("observed_mass"),
        )

    fragment_masses = fragment_masses.with_row_index("fragment_index")
    print(
        fragment_masses.filter(pl.col("is_internal"))
        .get_column("fragment_index")
        .to_list()
    )

    # Copy each fragment for each unique breakage weights
    fragments = pl.concat(
        [
            fragment_masses.with_columns(
                (
                    pl.col("observed_mass") - (breakage_weight * dp_table.precision)
                ).alias("standard_unit_mass"),
                pl.lit(breakages[0]).alias("breakage"),
            )
            for (breakage_weight, breakages) in breakage_dict.items()
        ]
    )

    # Filter out all fragments without any explanations
    fragments = (
        fragments.with_columns(
            pl.struct("observed_mass", "standard_unit_mass")
            .map_elements(
                lambda x: is_valid_su_mass(
                    mass=x["standard_unit_mass"],
                    dp_table=dp_table,
                    threshold=dp_table.tolerance * x["observed_mass"],
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
                integer_masses=[mass.mass for mass in dp_table.masses],
                dp_table=dp_table,
                threshold=dp_table.tolerance * x["observed_mass"],
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

    # Write terminal fragments to file if file name is given
    if output_file_path is not None:
        fragments.write_csv(output_file_path, separator="\t")

    return fragments


def is_singleton(mass, integer_masses, dp_table, threshold=None):
    # Convert the target to an integer for easy operations
    target = int(round(mass / dp_table.precision, 0))

    # Set relative threshold if not given
    if threshold is None:
        threshold = dp_table.tolerance * mass

    # Convert the threshold to integer
    threshold = int(np.ceil(threshold / dp_table.precision))

    # Check whether a singleton mass could be found
    for value in range(target - threshold, target + threshold + 1):
        if value in integer_masses:
            return True
    return False
