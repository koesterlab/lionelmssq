from lionelmssq.mass_explanation import explain_mass
import polars as pl

from lionelmssq.masses import EXPLANATION_MASSES, UNIQUE_MASSES, MATCHING_THRESHOLD
from lionelmssq.masses import TOLERANCE
from lionelmssq.masses import ROUND_DECIMAL
from concurrent.futures import ThreadPoolExecutor


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
    ms1_mass_deviations_allowed=0.001,  # 0.01,
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

    is_start = []  # Start tag in an explanation but NOT end tag in the SAME explanation
    is_end = []  # End tag in an explanation but NOT start tag in the SAME explanation
    is_start_end = []  # Both start and end tag in the same explanation (WITH OR WITHOUT MS1 MASS)
    is_internal = []  # NO tags in an explanation
    skip_mass = []
    # nucleotide_only_masses = []
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

    def process_mass(mass):
        explained_mass = explain_mass(
            mass, explanation_masses, matching_threshold=matching_threshold
        )

        if ms1_mass:
            explained_mass.explanations = {
                explanation
                for explanation in explained_mass.explanations
                if counts_subset(explanation, ms1_explained_mass.explanations)
            }

            if abs(mass / ms1_mass - 1) > ms1_mass_deviations_allowed:
                explained_mass.explanations = {
                    explanation
                    for explanation in explained_mass.explanations
                    if not ("3Tag" in explanation and "5Tag" in explanation)
                }
        else:
            explained_mass.explanations = {
                explanation
                for explanation in explained_mass.explanations
                if explanation.count("3Tag") <= 1 and explanation.count("5Tag") <= 1
            }

        result = {
            "mass_explanations": str(explained_mass.explanations),
            "singleton_mass": False,
            "skip_mass": False,
            "is_start_end": False,
            "is_start": False,
            "is_end": False,
            "is_internal": False,
        }

        if explained_mass.explanations != set():
            if any(len(element) == 1 for element in explained_mass.explanations):
                result["singleton_mass"] = True

            if all(
                element == ("3Tag",) for element in explained_mass.explanations
            ) or all(element == ("5Tag",) for element in explained_mass.explanations):
                result["skip_mass"] = True
            elif all(
                element == ("3Tag", "5Tag") for element in explained_mass.explanations
            ):
                result["skip_mass"] = True

            if ms1_mass:
                if abs(mass / ms1_mass - 1) < ms1_mass_deviations_allowed and any(
                    "5Tag" in element and "3Tag" in element
                    for element in explained_mass.explanations
                ):
                    result["is_start_end"] = True
            else:
                if any(
                    "5Tag" in element and "3Tag" in element
                    for element in explained_mass.explanations
                ):
                    result["is_start_end"] = True

            if not result["is_start_end"]:
                if any(
                    "5Tag" in element and "3Tag" not in element
                    for element in explained_mass.explanations
                ):
                    result["is_start"] = True

                if any(
                    "5Tag" not in element and "3Tag" in element
                    for element in explained_mass.explanations
                ):
                    result["is_end"] = True

                if any(
                    "5Tag" not in element and "3Tag" not in element
                    for element in explained_mass.explanations
                ):
                    result["is_internal"] = True

        else:
            result["skip_mass"] = True

        return result

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_mass, neutral_masses))

    for result in results:
        mass_explanations.append(result["mass_explanations"])
        singleton_mass.append(result["singleton_mass"])
        skip_mass.append(result["skip_mass"])
        is_start_end.append(result["is_start_end"])
        is_start.append(result["is_start"])
        is_end.append(result["is_end"])
        is_internal.append(result["is_internal"])

    # Use ms1_mass additionally as a cutoff for the fragment masses!
    if ms1_mass:
        mass_cutoff = (1.0 + ms1_mass_deviations_allowed) * ms1_mass  # 1.01 * ms1_mass

    fragment_masses = (
        fragment_masses.with_columns(
            # pl.Series(nucleotide_only_masses).alias(output_mass_column_name)
            pl.Series(neutral_masses).alias(output_mass_column_name)
        )
        .hstack(
            pl.DataFrame(
                {
                    "is_start": is_start,
                    "is_end": is_end,
                    "single_nucleoside": singleton_mass,
                    "is_start_end": is_start_end,
                    "is_internal": is_internal,
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
            print(
                "Mass = ",
                mass,
                "Closest mass = ",
                closest_mass,
                "Relative error = ",
                abs(closest_mass / mass),
            )

    if relative_errors:
        print("Relative errors = ", relative_errors)
        average_error = sum(relative_errors) / len(relative_errors)
        std_deviation = (
            sum((x - average_error) ** 2 for x in relative_errors)
            / len(relative_errors)
        ) ** 0.5
        return max(relative_errors), average_error, std_deviation
    else:
        return None, None, None
