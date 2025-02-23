from lionelmssq.mass_explanation import explain_mass
import polars as pl

from lionelmssq.masses import EXPLANATION_MASSES, UNIQUE_MASSES, MATCHING_THRESHOLD
from lionelmssq.masses import TOLERANCE
from lionelmssq.masses import ROUND_DECIMAL


def determine_terminal_fragments(
    fragment_masses,
    # fragment_masses_filepath,
    output_file_path=None,
    label_mass_3T=0.0,
    label_mass_5T=0.0,
    explanation_masses=EXPLANATION_MASSES,
    mass_column_name="neutral_mass",
    output_mass_column_name="observed_mass",
    intensity_cutoff=0.5e6,
    mass_cutoff=100000,
    matching_threshold=MATCHING_THRESHOLD
):
    # fragment_masses = pl.read_csv(fragment_masses_filepath, separator="\t")
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
    skip_mass = []
    nucleotide_only_masses = []
    mass_explanations = []
    singleton_mass = []

    for mass in neutral_masses:
        explained_mass = explain_mass(mass, explanation_masses, matching_threshold=matching_threshold)

        # Remove explainations which have more than one tag of each kind in them!
        # This greatly increases the reliability of tag determination!
        explained_mass.explanations = {
            explanation
            for explanation in explained_mass.explanations
            if explanation.count("3Tag") <= 1 and explanation.count("5Tag") <= 1
        }

        mass_explanations.append(str(explained_mass.explanations))

        if explained_mass.explanations != set():
            # print(mass, explained_mass.explanations)

            temp_list = []
            for element in explained_mass.explanations:
                temp_list.extend(element)

            # #Determine if its only a single nucleotide mass!
            if any(len(element) == 1 for element in explained_mass.explanations): #Only the case without tags is considered. Note: Check if to use all or any here!
                singleton_mass.append(True)
            else:
                singleton_mass.append(False)

            # #Determine if its only a single nucleotide mass! Also considers Tags with a single nucleotide as singleton!
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
            if set(temp_list) == {"3Tag"} or set(temp_list) == {"5Tag"}:
                skip_mass.append(True)
            elif set(temp_list) == {"3Tag", "5Tag"}:
                skip_mass.append(True)
            else:
                skip_mass.append(False)

            if "5Tag" in temp_list:
            #if "5Tag" in temp_list and len(explained_mass.explanations) == 1:
                nucleotide_only_masses.append(mass - label_mass_5T)
                is_start.append(True)
                is_end.append(False)
            elif "3Tag" in temp_list:
            #elif "3Tag" in temp_list and len(explained_mass.explanations) == 1:
                nucleotide_only_masses.append(mass - label_mass_3T)
                is_end.append(True)
                is_start.append(False)
            else:
                nucleotide_only_masses.append(mass)
                is_start.append(False)
                is_end.append(False)
        else:
            nucleotide_only_masses.append(mass)
            skip_mass.append(True)
            is_start.append(False)
            is_end.append(False)
            singleton_mass.append(False)

    # TODO: Determine the fragments with both of the tags intact and output is_start = True and is_end = True! That will be the full sequence!
    # We haven't thrown away the case where the two different types of tags can be present!

    fragment_masses = (
        fragment_masses.with_columns(
            pl.Series(nucleotide_only_masses).alias(output_mass_column_name)
        )
        .hstack(pl.DataFrame({"is_start": is_start, "is_end": is_end, "single_nucleoside": singleton_mass}))
        .with_columns(pl.Series(mass_explanations).alias("mass_explanations"))
        .filter(~pl.Series(skip_mass))
        # .filter(
        #    pl.col("neutral_mass") > 305.04129
        # )
        .sort(pl.col("observed_mass"))
        .filter(pl.col("intensity") > intensity_cutoff)
        .filter(
            pl.col("neutral_mass") < mass_cutoff
        )  # TODO: Replace this by an estimate of the max mass of the sequence!
    )

    if output_file_path is not None:
        fragment_masses.write_csv(output_file_path, separator="\t")

    return fragment_masses

def estimate_MS_error_MATCHING_THRESHOLD(fragments,unique_masses=UNIQUE_MASSES,rejection_threshold=0.5,simulation=False):
    """
    Using the mass of the single nucleosides, A, U, G, C, estimate the relatve error that the MS makes, this is used to determine the MATCHING_THRESHOLD for the DP algorithm!

    """

    unique_natural_masses = unique_masses.filter(
            pl.col("nucleoside").is_in(["A", "U", "G", "C"])
        ).select(pl.col("monoisotopic_mass")).to_series().to_list()
    

    if simulation:
        singleton_masses = fragments.filter(pl.col("single_nucleoside")).select(pl.col("observed_mass")).to_series().to_list()
    else:
        singleton_masses = fragments.filter(pl.col("neutral_mass").is_between(min(unique_natural_masses)-rejection_threshold,max(unique_natural_masses)+rejection_threshold)).select(pl.col("neutral_mass")).to_series().to_list()

    relative_errors = []
    for mass in singleton_masses:
        differences = [abs(unique_mass - mass) for unique_mass in unique_natural_masses]
        closest_mass = min(differences) if min(differences) < rejection_threshold else None
        if closest_mass:
            relative_errors.append(abs(closest_mass/mass))

    if relative_errors:
        average_error = sum(relative_errors) / len(relative_errors)
        std_deviation = (sum((x - average_error) ** 2 for x in relative_errors) / len(relative_errors)) ** 0.5
        return max(relative_errors),average_error, std_deviation
    else:
        return None, None, None