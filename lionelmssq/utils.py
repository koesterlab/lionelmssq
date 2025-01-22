from lionelmssq.mass_explanation import explain_mass
import polars as pl
import re

from lionelmssq.masses import EXPLANATION_MASSES
from lionelmssq.masses import TOLERANCE
from lionelmssq.masses import ROUND_DECIMAL


def determine_terminal_fragments(
    fragment_masses_filepath,
    output_file_path=None,
    label_mass_3T=0.0,
    label_mass_5T=0.0,
    explanation_masses=EXPLANATION_MASSES,
    mass_column_name="neutral_mass",
    output_mass_column_name="observed_mass",
    intensity_cutoff=0.5e6,
):
    fragment_masses = pl.read_csv(fragment_masses_filepath, separator="\t")
    neutral_masses = (
        fragment_masses.select(pl.col(mass_column_name)).to_series().to_list()
    )

    # Inferred from Shanice's RNA file!
    # DNA
    # label_mass_3T = 455.14912 #3' label  #y-fragments
    # label_mass_5T = 635.15565 #5' label #c-fragments

    # regex for separating given sequence into nucleosides
    # nucleoside_re5T = re.compile(r"\d*[5T]")
    # nucleoside_re3T = re.compile(r"\d*[3T]")

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

    for mass in neutral_masses:
        explained_mass = explain_mass(mass, explanation_masses)

        if explained_mass.explanations != set():
            temp_list = []
            for element in explained_mass.explanations:
                temp_list.extend(element)
            # temp_list = "".join(temp_list)

            # Do not consider the mass if it is purely only explained by the tags!
            if set(temp_list) == {"3Tag"} or set(temp_list) == {"5Tag"}:
                skip_mass.append(True)
            else:
                skip_mass.append(False)

            # TODO: Only output if a sequence is tagged IF all possible DP solutions of the sequence have the tag in there!
            # Can also restrict this by doing this above a threshold value!

            # if '5T' in nucleoside_re5T.findall(temp_list):
            if "5Tag" in temp_list:
                nucleotide_only_masses.append(mass - label_mass_5T)
                is_start.append(True)
                is_end.append(False)
            # elif '3T' in nucleoside_re3T.findall(temp_list):
            elif "3Tag" in temp_list:
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

    # TODO: Determine the fragments with both of the tags intact and output is_start = True and is_end = True!

    fragment_masses = (
        fragment_masses.with_columns(
            pl.Series(nucleotide_only_masses).alias(output_mass_column_name)
        )
        .hstack(pl.DataFrame({"is_start": is_start, "is_end": is_end}))
        .filter(~pl.Series(skip_mass))
        .filter(
            pl.col("neutral_mass") > 305.04129
        )  # TODO: Replace this by min of nucleotides or nucleotides with tags etc.
        .sort(pl.col("observed_mass"))
        .filter(pl.col("intensity") > intensity_cutoff)
    )
    # .filter(pl.col("neutral_mass") < 5500 ) #TODO: Replace this by an estimate of the max mass of the sequence!

    if output_file_path is not None:
        fragment_masses.write_csv(output_file_path, separator="\t")

    return fragment_masses
