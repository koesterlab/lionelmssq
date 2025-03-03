from lionelmssq.mass_explanation import explain_mass
import polars as pl

from lionelmssq.masses import EXPLANATION_MASSES, UNIQUE_MASSES, MATCHING_THRESHOLD
from lionelmssq.masses import TOLERANCE
from lionelmssq.masses import ROUND_DECIMAL


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


def determine_terminal_fragments(
    fragment_masses,
    ms1_mass=None,
    output_file_path=None,
    label_mass_3T=0.0,
    label_mass_5T=0.0,
    explanation_masses=EXPLANATION_MASSES,
    mass_column_name="neutral_mass",
    output_mass_column_name="observed_mass",
    intensity_cutoff=0.5e6,
    mass_cutoff=50000,
    matching_threshold=MATCHING_THRESHOLD,
):
    neutral_masses = (
        fragment_masses.select(pl.col(mass_column_name)).to_series().to_list()
    )

    tags = ["3Tag", "5Tag"]
    tag_masses = [
        round(label_mass_3T, ROUND_DECIMAL),
        round(label_mass_5T, ROUND_DECIMAL),
    ]
    nucleoside_df = pl.DataFrame({"nucleoside": tags, "monoisotopic_mass": tag_masses})

    nucleoside_df = nucleoside_df.with_columns(
        (pl.col("monoisotopic_mass") / TOLERANCE)
        .round(0)
        .cast(pl.Int64)
        .alias("tolerated_integer_masses")
    )

    explanation_masses = explanation_masses.vstack(nucleoside_df)

    is_start = []
    is_end = []
    is_start_end = [] #Also output a column which says if there are independent 3T and 5T explanations. We can start with one kind of explanation while skelelton building and if that fragemnet is rejected, we can try to add it to the "end" explanation!
    skip_mass = []
    nucleotide_only_masses = []
    mass_explanations = []
    singleton_mass = []

    if ms1_mass:
        ms1_explained_mass = explain_mass(
            ms1_mass, explanation_masses, matching_threshold=matching_threshold
        )
        ms1_explained_mass.explanations = {
            element
            for element in ms1_explained_mass.explanations
            if element.count("3Tag") == 1 and element.count("5Tag") == 1
        }
        # print(ms1_explained_mass.explanations)
        if not ms1_mass:
            print("MS1 mass is not explained by the nucleosides!")
            ms1_mass = None

    for mass in neutral_masses:
        explained_mass = explain_mass(
            mass, explanation_masses, matching_threshold=matching_threshold
        )

        if ms1_mass:
            explained_mass.explanations = {
                explanation
                for explanation in explained_mass.explanations
                if counts_subset(explanation, ms1_explained_mass.explanations)
                # if the counts of the different nucleosides are less than or equal to the respective counts in the MS1 mass explanation
            }
            # print(explained_mass.explanations)

            #If ms1 mass is defined, then also remove the explanations which differ in mass by more than 1% and have both kind of tags in there!
            if abs(mass / ms1_mass - 1) > 0.01:
                explained_mass.explanations = {
                    explanation
                    for explanation in explained_mass.explanations
                    if not ("3Tag" in explanation and "5Tag" in explanation)
                }
        else:
            # Remove explainations which have more than one tag of each kind in them!
            # This greatly increases the reliability of tag determination!
            explained_mass.explanations = {
                explanation
                for explanation in explained_mass.explanations
                if explanation.count("3Tag") <= 1 and explanation.count("5Tag") <= 1
            }

        mass_explanations.append(str(explained_mass.explanations))

        if explained_mass.explanations != set():
            # #Determine if its only a single nucleotide mass!
            if any(
                len(element) == 1 for element in explained_mass.explanations
            ):  # Only the case without tags is considered. Note: Check if to use all or any here!
                singleton_mass.append(True)
            else:
                singleton_mass.append(False)

            # #Determine if its only a single nucleotide mass! The following code also considers Tags with a single nucleotide as singleton!
            # singleton_bool = False
            # for element in explained_mass.explanations:
            #     if len(element) == 1:
            #         singleton_mass.append(True)
            #         singleton_bool = True
            #         break
            #     elif len(element) == 2:
            #         if "3Tag" in element or "5Tag" in element:
            #             singleton_mass.append(True)
            #             singleton_bool = True
            #             break
            # if not singleton_bool:
            #     singleton_mass.append(False)

            # Do not consider the mass if it is purely only explained by the tags!
            # This is slightly redundant with earlier pruning based count of tags, but ensures that we are not trying to fit fragments with only tags!
            if all(
                element == ("3Tag",) for element in explained_mass.explanations
            ) or all(element == ("5Tag",) for element in explained_mass.explanations):
                skip_mass.append(True)
            elif all(
                element
                == (
                    "3Tag",
                    "5Tag",
                )
                for element in explained_mass.explanations
            ):
                skip_mass.append(True)
            else:
                skip_mass.append(False)

            if ms1_mass:
                if (
                    abs(mass / ms1_mass - 1) < 0.01
                    and any(
                        "5Tag" in element and "3Tag" in element
                        for element in explained_mass.explanations
                    )
                ):  # Note that we allow for approx 1% deviation from the MS1 mass for both tags!
                    # TODO: This still first preferrentially marks this case, and then 5Tag and then 3Tag!
                    nucleotide_only_masses.append(mass - label_mass_3T - label_mass_5T)
                    is_start.append(True)
                    is_end.append(True)
                    is_start_end.append(False)
                elif any(
                    "5Tag" in element and "3Tag" not in element
                    for element in explained_mass.explanations
                ):
                    nucleotide_only_masses.append(mass - label_mass_5T)
                    is_start.append(True)
                    is_end.append(False)
                    if any(
                        "3Tag" in element and "5Tag" not in element
                        for element in explained_mass.explanations
                    ):
                        is_start_end.append(True)
                    else:
                        is_start_end.append(False)
                elif any(
                    "5Tag" not in element and "3Tag" in element
                    for element in explained_mass.explanations
                ):
                    nucleotide_only_masses.append(mass - label_mass_3T)
                    is_end.append(True)
                    is_start.append(False)
                    is_start_end.append(False)
                else:
                    nucleotide_only_masses.append(mass)
                    is_start.append(False)
                    is_end.append(False)
                    is_start_end.append(False)
            else:
                if any(
                    "5Tag" in element and "3Tag" not in element
                    for element in explained_mass.explanations
                ):
                    nucleotide_only_masses.append(mass - label_mass_5T)
                    is_start.append(True)
                    is_end.append(False)
                    if any(
                        "3Tag" in element and "5Tag" not in element
                        for element in explained_mass.explanations
                    ):
                        is_start_end.append(True)
                    else:
                        is_start_end.append(False)
                elif any(
                    "5Tag" not in element and "3Tag" in element
                    for element in explained_mass.explanations
                ):
                    nucleotide_only_masses.append(mass - label_mass_3T)
                    is_end.append(True)
                    is_start.append(False)
                    is_start_end.append(False)
                else:
                    nucleotide_only_masses.append(mass)
                    is_start.append(False)
                    is_end.append(False)
                    is_start_end.append(False)

        else:
            nucleotide_only_masses.append(mass)
            skip_mass.append(True)
            is_start.append(False)
            is_end.append(False)
            singleton_mass.append(False)
            is_start_end.append(False)

    # Use ms1_mass additionally as a cutoff for the fragment masses!
    if ms1_mass:
        mass_cutoff = 1.01 * ms1_mass

    fragment_masses = (
        fragment_masses.with_columns(
            pl.Series(nucleotide_only_masses).alias(output_mass_column_name)
        )
        .hstack(
            pl.DataFrame(
                {
                    "is_start": is_start,
                    "is_end": is_end,
                    "single_nucleoside": singleton_mass,
                    "is_start_end": is_start_end,
                }
            )
        )
        .with_columns(pl.Series(mass_explanations).alias("mass_explanations"))
        .filter(~pl.Series(skip_mass))
        # .filter(
        #    pl.col("neutral_mass") > 305.04129
        # )
        .sort(pl.col("observed_mass"))
        .filter(pl.col("intensity") > intensity_cutoff)
        .filter(pl.col("neutral_mass") < mass_cutoff)
    )

    if output_file_path is not None:
        fragment_masses.write_csv(output_file_path, separator="\t")

    return fragment_masses


def estimate_MS_error_MATCHING_THRESHOLD(
    fragments, unique_masses=UNIQUE_MASSES, rejection_threshold=0.5, simulation=False
):
    """
    Using the mass of the single nucleosides, A, U, G, C, estimate the relatve error that the MS makes, this is used to determine the MATCHING_THRESHOLD for the DP algorithm!

    """

    unique_natural_masses = (
        unique_masses.filter(pl.col("nucleoside").is_in(["A", "U", "G", "C"]))
        .select(pl.col("monoisotopic_mass"))
        .to_series()
        .to_list()
    )

    if simulation:
        singleton_masses = (
            fragments.filter(pl.col("single_nucleoside"))
            .select(pl.col("observed_mass"))
            .to_series()
            .to_list()
        )
    else:
        singleton_masses = (
            fragments.filter(
                pl.col("neutral_mass").is_between(
                    min(unique_natural_masses) - rejection_threshold,
                    max(unique_natural_masses) + rejection_threshold,
                )
            )
            .select(pl.col("neutral_mass"))
            .to_series()
            .to_list()
        )

    relative_errors = []
    for mass in singleton_masses:
        differences = [abs(unique_mass - mass) for unique_mass in unique_natural_masses]
        closest_mass = (
            min(differences) if min(differences) < rejection_threshold else None
        )
        if closest_mass:
            relative_errors.append(abs(closest_mass / mass))

    if relative_errors:
        average_error = sum(relative_errors) / len(relative_errors)
        std_deviation = (
            sum((x - average_error) ** 2 for x in relative_errors)
            / len(relative_errors)
        ) ** 0.5
        return max(relative_errors), average_error, std_deviation
    else:
        return None, None, None
