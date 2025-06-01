from lionelmssq.mass_explanation import explain_mass
import polars as pl
from collections import Counter

from lionelmssq.masses import EXPLANATION_MASSES, UNIQUE_MASSES, MATCHING_THRESHOLD
from lionelmssq.masses import TOLERANCE
from lionelmssq.masses import ROUND_DECIMAL
from lionelmssq.prediction import Predictor


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
        if not ms1_explained_mass.explanations:
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

            # If ms1 mass is defined, then also remove the explanations which differ in mass by more than 1% and have both kind of tags in there!
            if abs(mass / ms1_mass - 1) > ms1_mass_deviations_allowed:
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
                    abs(mass / ms1_mass - 1) < ms1_mass_deviations_allowed
                    and any(
                        "5Tag" in element and "3Tag" in element
                        for element in explained_mass.explanations
                    )
                ):  # Note that we allow for approx 1% deviation from the MS1 mass for both tags!
                    # TODO: This still first preferrentially marks this case, and then 5Tag and then 3Tag!
                    is_start_end.append(True)
                else:
                    is_start_end.append(False)
            else:
                if any(
                    "5Tag" in element and "3Tag" in element
                    for element in explained_mass.explanations
                ):
                    is_start_end.append(True)
                else:
                    is_start_end.append(False)

            if not is_start_end[-1]:
                if any(
                    "5Tag" in element and "3Tag" not in element
                    for element in explained_mass.explanations
                ):
                    is_start.append(True)
                else:
                    is_start.append(False)

                if any(
                    "5Tag" not in element and "3Tag" in element
                    for element in explained_mass.explanations
                ):
                    is_end.append(True)
                else:
                    is_end.append(False)

                if any(
                    "5Tag" not in element and "3Tag" not in element
                    for element in explained_mass.explanations
                ):
                    is_internal.append(True)
                else:
                    is_internal.append(False)

            else:
                is_start.append(False)
                is_end.append(False)
                is_internal.append(False)

        else:
            # nucleotide_only_masses.append(mass)
            skip_mass.append(True)
            is_start.append(False)
            is_end.append(False)
            singleton_mass.append(False)
            is_start_end.append(False)
            is_internal.append(False)

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
        .sort(pl.col("observed_mass"))
        .filter(pl.col("intensity") > intensity_cutoff)
        .filter(pl.col("neutral_mass") < mass_cutoff)
    )

    if output_file_path is not None:
        fragment_masses.write_csv(output_file_path, separator="\t")

    return fragment_masses


def predetermine_possible_nucleotides(
    fragments,
    singleton_mass_filtering_limit=1.1
    * (
        max(UNIQUE_MASSES.select(pl.col("monoisotopic_mass")).to_series().to_list())
        + 61.95577
    ),
    matching_threshold=2e-5,
    explanation_masses=EXPLANATION_MASSES,
    intensity_cutoff=0.5e6,
):
    singleton_masses = (
        fragments.filter(pl.col("neutral_mass") < singleton_mass_filtering_limit)
        .filter(pl.col("intensity") > intensity_cutoff)
        .select(pl.col("neutral_mass"))
        .to_series()
        .to_list()
    )

    explanations = {}

    for mass in singleton_masses:
        expl = explain_mass(
            mass, explanation_masses, matching_threshold=matching_threshold
        ).explanations
        explanations[mass] = expl

    observed_nucleosides = {
        nuc
        for expls in explanations.values()
        for expl in expls
        for nuc in expl
        if len(expl) == 1
    }

    reduced = explanation_masses.filter(
        pl.col("nucleoside").is_in(observed_nucleosides)
    )

    nucleosides = reduced.get_column("nucleoside").to_list()

    return nucleosides


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
        average_error = sum(relative_errors) / len(relative_errors)
        std_deviation = (
            sum((x - average_error) ** 2 for x in relative_errors)
            / len(relative_errors)
        ) ** 0.5
        return max(relative_errors), average_error, std_deviation
    else:
        return None, None, None


def determine_sequence_length(
    terminally_marked_fragments=None,
    strategy="entropy",  # "entropy" or "frequency"
    ms1_mass=None,
    label_mass_3T=0.0,
    label_mass_5T=0.0,
    explanation_masses=EXPLANATION_MASSES,
    matching_threshold=MATCHING_THRESHOLD,
    ms1_mass_deviations_allowed=0.001,
):
    """
    Determine the sequence length of the input sequence.
    #TODO: Implement the direct computaton without passing terminally_marked_fragments!
    """

    def _calculate_entropy(sequence):
        """
        Calculate the entropy of a sequence.
        """
        from collections import Counter
        from math import log2

        counts = Counter(sequence)
        probabilities = [count / len(sequence) for count in counts.values()]

        return -sum(p * log2(p) for p in probabilities if p > 0)

    if terminally_marked_fragments is not None:
        # Check if the terminally marked fragments are already determined
        if terminally_marked_fragments.is_empty():
            raise ValueError("The terminally marked fragments are empty.")
        else:
            start_end_fragments_series = (
                terminally_marked_fragments.filter(pl.col("is_start_end"))
                .select(pl.col("mass_explanations"))
                .to_series()
                .to_list()
            )

            sequence_explanations = []
            for explanations in start_end_fragments_series:
                for explanation in eval(explanations):
                    sequence_explanations.append(list(explanation))

            len_sequence_explanations = [
                (len(explanations) - 2)
                for explanations in sequence_explanations
                if "3Tag" in explanations and "5Tag" in explanations
            ]

            entropy_sequence_explanations = [
                _calculate_entropy(explanations)
                for explanations in sequence_explanations
            ]

            length_frequency_dict = Counter(len_sequence_explanations)

            frequency_sequence_explanations = [
                length_frequency_dict[len_explanation]
                for len_explanation in len_sequence_explanations
            ]

            if strategy == "entropy":
                # Order sequence_explanations and their lengths according to entropy and then according to frequency and then according to length

                ordered_data = sorted(
                    zip(
                        sequence_explanations,
                        len_sequence_explanations,
                        entropy_sequence_explanations,
                        frequency_sequence_explanations,
                    ),
                    key=lambda x: (x[2], x[3], x[1]),  # Sort by entropy, then by length
                    reverse=True,  # Sort by entropy
                )

                ordered_sequence_explanations = [data[0] for data in ordered_data]
                ordered_lengths = [data[1] for data in ordered_data]
                ordered_strategy = [data[2] for data in ordered_data]

            elif strategy == "frequency":
                # Order sequence_explanations and their lengths according to frequency

                ordered_data = sorted(
                    zip(
                        sequence_explanations,
                        len_sequence_explanations,
                        frequency_sequence_explanations,
                        entropy_sequence_explanations,
                    ),
                    key=lambda x: (
                        x[2],
                        x[3],
                        x[1],
                    ),  # Sort by frequency, then by entropy and then by length
                    reverse=True,  # Sort by frequency and length
                )

                ordered_sequence_explanations = [data[0] for data in ordered_data]
                ordered_lengths = [data[1] for data in ordered_data]
                ordered_strategy = [data[2] for data in ordered_data]

            return ordered_lengths, ordered_strategy, ordered_sequence_explanations
