import polars as pl
import numpy as np

from lionelmssq.masses import (
    MATCHING_THRESHOLD,
    TOLERANCE,
    BREAKAGES,
)
from lionelmssq.mass_explanation import (
    is_valid_mass,
    # explain_mass_with_dp,
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
    return is_valid_mass(mass, dp_table, breakages)


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
    return is_valid_mass(mass, dp_table, breakages)


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
    return is_valid_mass(mass, dp_table, breakages)


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
    return is_valid_mass(mass, dp_table, breakages)


def is_singleton_candidate(mass, integer_masses, threshold=MATCHING_THRESHOLD):
    # Convert the target to an integer for easy operations
    target = int(round(mass / TOLERANCE, 0))

    # Set matching threshold based on target mass
    threshold = int(np.ceil(threshold * target))

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
    # threshold=MATCHING_THRESHOLD,
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
            lambda x: is_start_fragment_candidate(x, dp_table=dp_table.table),
            return_dtype=bool,
        )
        .alias("is_start")
    )

    # Determine all fragments that may be terminal ones (end only)
    fragments = fragments.with_columns(
        pl.col(mass_column_name)
        .map_elements(
            lambda x: is_end_fragment_candidate(x, dp_table=dp_table.table),
            return_dtype=bool,
        )
        .alias("is_end")
    )

    # Determine all fragments that may be singletons (with or without tags)
    fragments = fragments.with_columns(
        pl.col(mass_column_name)
        .map_elements(
            lambda x: is_singleton_candidate(x, integer_masses=integer_masses),
            return_dtype=bool,
        )
        .alias("single_nucleoside")
    )

    # Determine all fragments that may be the complete sequence
    fragments = fragments.with_columns(
        pl.col(mass_column_name)
        .map_elements(
            lambda x: is_complete_fragment_candidate(x, dp_table=dp_table.table),
            return_dtype=bool,
        )
        .alias("is_start_end")
    )

    # Determine all fragments that may be internal ones
    fragments = fragments.with_columns(
        pl.col(mass_column_name)
        .map_elements(
            lambda x: is_internal_fragment_candidate(x, dp_table=dp_table.table),
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
        .filter(pl.col("neutral_mass") < mass_cutoff)
    )

    # Write terminal fragments to file if file name is given
    if output_file_path is not None:
        fragments.write_csv(output_file_path, separator="\t")

    return fragments
