# %%
import os
from pathlib import Path

import altair as alt
import numpy as np
import polars as pl
import tqdm as tqdm
import yaml
from Bio import SeqIO
from loguru import logger
from pyxdameraulevenshtein import normalized_damerau_levenshtein_distance_seqs

import spectrseqtools.multiplexing as multiplexing
from spectrseqtools.dataclasses import Sequence
from spectrseqtools.enums import SolverType
from spectrseqtools.error_calculator import ErrorCalculator
from spectrseqtools.file_settings import DEFAULT_ALPHABET_PATH
from spectrseqtools.nucleotide_alphabet import NucleotideAlphabet
from spectrseqtools.parsers import Options, PredictionOptions, PreprocessingOptions
from spectrseqtools.prediction.prediction import Predictor

# %%
# REFERENCE SEQUENCE INPUT

# Version that takes in a list of sequences
# seq_list = [
#     "GAUUGUCGUG",
#     "GCGUAUCCGGCGAGGUGACG",
#     "AUGAUGAUGAUGAUGAUGAU",
#     "UGCCUCACCACIUCAUCCUG",
#     "UGCCAUUAUACAUGA",
#     "CCAʍGUGCUAAJAAAUUGCC",
#     "UUUCAAGGGAGUUACUUCAG",
#     "UUAGCCCG＃A",
#     "GAUBUAGAUC",
#     "UAGCGUUUGAUGAUC"
#             ]
# target_record = "".join(seq_list)

# Version that takes in a fastafile from MODOMICS
rna_subtype = "Gly"
rna_modomics_id = "tdbR00000117"
fastafile = SeqIO.parse(
    "C:\\Users\\juan\\Documents\\Postdoc\\modified_tRNA_Gly_all_rna_sequences.fasta",
    "fasta",
)
for record in fastafile:
    record_id = dict([item.split(":", 1) for item in record.id.split("|")])
    if record_id["Subtype"] == rna_subtype and record_id["Name"] == rna_modomics_id:
        target_record = record

# Standardize reference sequence
NUCLEOTIDE_DF = NucleotideAlphabet.from_file(
    ErrorCalculator.with_metric()
).to_dataframe()
masses = pl.read_csv(DEFAULT_ALPHABET_PATH, separator="\t")


def reference_sequence(sequence_string):
    target_sequence = []
    for n in list(sequence_string):
        if n == " ":
            target_sequence.append([""])
            continue
        encoding_to_id = masses.filter(pl.col("encoding") == n).select("id").item()
        id_list = (
            NUCLEOTIDE_DF.filter(pl.col("names").list.contains(encoding_to_id))
            .select("names")
            .item()
            .to_list()
        )
        target_sequence.append(id_list)

    return target_sequence


# Standardized reference sequence to be used in alignment at the last step
target_sequence = reference_sequence(target_record)

# %%
# INPUT PARAMETERS AND FILE PATH TO RAW FILE
three_prime_tag = 728.2006
five_prime_tag = 170.9755
start_time = 10.0
end_time = 15.0
delta_time_window = 0.2
intensity_cutoff_percentile = 50
file_path = Path(r"endlabeled_trna\2025-12-03_SJH_tRNAGly_7ug_fullworkflow_02.raw")

# %%
# PARSERS
sample_name = file_path.stem
options = Options(
    preprocessing=PreprocessingOptions(
        input=file_path,
        meta=None,
        output_dir=f"output/{sample_name}/preprocessing/",
        alphabet=None,
        ms1_charge_range=None,
        ms2_charge_range=None,
        min_intensity=None,
    ),
    prediction=PredictionOptions(
        fragments=Path(""),  # Dummy path
        meta=Path(f"output/{sample_name}/preprocessing/preprocessed.meta.yaml"),
        alphabet=Path(""),  # Dummy path
        solver=SolverType.GUROBI,
        fragment_predictions=Path(
            f"output/{sample_name}/prediction/fragments.prediction.tsv"
        ),
        sequence_prediction=Path(
            f"output/{sample_name}/prediction/fragments.prediction.fasta"
        ),
        sequence_name="",  # Dummy path
        intensity_cutoff_percentile=intensity_cutoff_percentile,
        output_dir=f"output/{sample_name}/prediction/",
    ),
    postprocessing=None,
    plotting=None,
)
# %%
# PRE-PROCESSING CELL
(fragments, singletons, meta) = multiplexing.pre_process_multiplexing(
    options=options,
    three_prime_tag=three_prime_tag,
    five_prime_tag=five_prime_tag,
    delta_time_window=delta_time_window,
    start_time=start_time,
    end_time=end_time,
)

# Save all pre-processing files in preprocessing.output_dir
if not os.path.exists(options.preprocessing.output_dir):
    os.makedirs(options.preprocessing.output_dir)

singletons.write_csv(
    str(options.preprocessing.output_dir) + "preprocessed.singletons.tsv",
    separator="\t",
)
fragments.write_csv(
    str(options.preprocessing.output_dir) + "preprocessed.fragments.tsv", separator="\t"
)
with open(
    str(options.preprocessing.output_dir) + "preprocessed.meta.yaml",
    "w",
    encoding="utf-8",
) as f:
    yaml.dump(meta, f)

# %%
# PREDICTION CELL

# Load fragments and singletons
fragments = pl.read_csv(
    str(options.preprocessing.output_dir) + "preprocessed.fragments.tsv", separator="\t"
)
singletons = pl.read_csv(
    str(options.preprocessing.output_dir) + "preprocessed.singletons.tsv",
    separator="\t",
)

# Generate list of MS1 group numbers (number of peaks with fragments)
grp_number_fragments = set(fragments["ms1_mass_group"].unique().to_list())
grp_number_singletons = set(singletons["ms1_mass_group"].unique().to_list())
# Check if fragments and singletons contain the same group numbers
if grp_number_fragments == grp_number_singletons:
    grp_numbers = grp_number_fragments
else:
    raise Exception("MS1 mass groups in fragments and singletons are not equal!")

# Main prediction loop
all_raw_fragments = []
all_prediction_fragments = []
all_fasta_dicts = {}

for grp_number in tqdm.tqdm(grp_numbers, desc="Performing prediction"):
    fragments_i = fragments.filter(pl.col("ms1_mass_group") == grp_number)
    alphabet_i = singletons.filter(pl.col("ms1_mass_group") == grp_number)

    # Update prediction options
    options.prediction.fragments = fragments_i
    options.prediction.alphabet = alphabet_i
    options.prediction.sequence_name = f"{sample_name}.{grp_number}"

    print(f"Group {grp_number}")

    # Main prediction function
    logger.disable("spectrseqtools.fragments")
    try:
        raw_fragments, prediction_fragments, fasta_dict = Predictor(
            options.prediction
        ).predict()
        prediction_fragments = prediction_fragments.with_columns(
            pl.lit(grp_number).alias("ms1_mass_group")
        )
    except NotImplementedError:
        continue

    all_raw_fragments.append(raw_fragments)
    all_prediction_fragments.append(prediction_fragments)
    all_fasta_dicts.update(fasta_dict)

# Collect and concatenate prediction fragments
raw_fragments = pl.concat(all_raw_fragments)
prediction_fragments = pl.concat(all_prediction_fragments)

# Save all prediction files in prediction.output_dir
if not os.path.exists(options.prediction.output_dir):
    os.makedirs(options.prediction.output_dir)

with open(str(options.prediction.sequence_prediction), "w") as f:
    for header, sequence in all_fasta_dicts.items():
        f.write(f"{header}\n")
        f.write(f"{sequence}\n")
raw_fragments.write_csv(
    str(options.prediction.output_dir)
    + "preprocessed.fragments.standard_unit_fragments.tsv",
    separator="\t",
)
prediction_fragments.write_csv(
    str(options.prediction.fragment_predictions), separator="\t"
)

# %%
# POST-PROCESSING AND ALIGNMENT

# Load predictions fasta file and generate sequence dictionary, mapping MS1 group number to prediction
sequence_dict = {}
with open(str(options.prediction.sequence_prediction), mode="r", encoding="utf-8") as f:
    lines = f.readlines()

    for line in zip(lines[::2], lines[1::2]):
        head, seq = line

        assert head.startswith(">")
        if head.endswith("_full\n"):
            continue

        grp_number = int(head.split(".")[-1].removesuffix("\n"))
        sequence = Sequence.from_str(seq)
        if len(sequence.sequence) == 0:
            continue
        sequence_dict[grp_number] = sequence

# Get list of intact masses
fragments = pl.read_csv(
    str(options.preprocessing.output_dir) + "preprocessed.fragments.tsv", separator="\t"
)
ms1_masses = fragments.filter(
    pl.col("is_ms1_mass") & pl.col("ms1_mass_group").is_in(list(sequence_dict.keys()))
).with_columns(
    pl.col("min_window_time").cast(pl.Float64),
    pl.col("max_window_time").cast(pl.Float64),
)

# Generate prediction list for alignment
prediction_vals = []

for grp_number, seq in sequence_dict.items():
    ms1_mass_info = ms1_masses.filter(pl.col("ms1_mass_group") == grp_number)
    assert len(ms1_mass_info) == 1
    ms1_mass_info = ms1_mass_info[0]

    prediction_vals.append(
        {
            "group_number": grp_number,
            "intact_mass": ms1_mass_info["observed_mass"][0],
            "predicted_sequence": seq.sequence,
            "min_window_time": ms1_mass_info["min_window_time"][0],
            "max_window_time": ms1_mass_info["max_window_time"][0],
            "adduct_types": ms1_mass_info["adduct_type"][0],
        }
    )


# Alignment and plotting functions
def targ_pred_pairing(x):
    if x[0] in x[1]:
        return (x[0], x[0])
    else:
        return (x[0], x[1][0])


def compare_prediction_to_target(prediction_raw, target_sequence, is_backward=False):
    if is_backward:
        prediction_raw = prediction_raw[::-1]
    prediction_len = len(prediction_raw)
    prediction_string = "".join(
        masses.filter(pl.col("id") == n).select("encoding").item()
        for n in prediction_raw
    )

    target_strings = []

    for i in range(len(target_sequence) - prediction_len + 1):
        target_sequence_window = target_sequence[i : i + prediction_len]
        targ_string = ""

        for p in zip(prediction_raw, target_sequence_window):
            pair = targ_pred_pairing(p)
            targ_string += masses.filter(pl.col("id") == pair[1])["encoding"].item()
        target_strings.append(targ_string)

    distances = normalized_damerau_levenshtein_distance_seqs(
        prediction_string, target_strings
    )
    min_distance_idx = np.argmin(distances)

    min_distance = distances[min_distance_idx]
    start_pos = min_distance_idx
    end_pos = min_distance_idx + prediction_len

    return (
        prediction_string,
        target_strings[min_distance_idx],
        min_distance,
        start_pos,
        end_pos,
        is_backward,
    )


def align_prediction_results(
    prediction_vals, target_sequence, alignment_score_threshold=0
):

    alignment_vals = []

    for i in tqdm.tqdm(prediction_vals, desc="Calculating alignment scores"):
        prediction_raw = i["predicted_sequence"]
        if len(prediction_raw) == 0:
            continue

        forward_comparison = compare_prediction_to_target(
            prediction_raw, target_sequence, is_backward=False
        )
        backward_comparison = compare_prediction_to_target(
            prediction_raw, target_sequence, is_backward=True
        )

        final_comparison = (
            forward_comparison
            if forward_comparison[2] <= backward_comparison[2]
            else backward_comparison
        )

        if final_comparison[2] > alignment_score_threshold:
            continue

        alignment_vals.append(
            final_comparison
            + (
                i["group_number"],
                i["intact_mass"],
                i["min_window_time"],
                i["max_window_time"],
                i["adduct_types"],
            )
        )

    df_alignment = pl.DataFrame(
        alignment_vals,
        schema=[
            "predicted_string",
            "best_matching_target_string",
            "normalized_damerau_levenshtein_distance",
            "target_start_pos",
            "target_end_pos",
            "is_backward",
            "group",
            "intact_mass",
            "min_window_time",
            "max_window_time",
            "adduct_type",
        ],
        orient="row",
    )

    return df_alignment


def interpret_alignment_results(df_alignment, target_sequence):

    match_rows = []

    for r in df_alignment.iter_rows(named=True):
        pred_string = r["predicted_string"]
        target_string = r["best_matching_target_string"]
        start_pos = r["target_start_pos"]
        intact_mass = r["intact_mass"]
        min_window_time = r["min_window_time"]
        max_window_time = r["max_window_time"]
        adduct_type = r["adduct_type"]

        for i, (p, t) in enumerate(zip(pred_string, target_string)):
            if p == t:
                status = "match"
            else:
                status = "mismatch"
            match_rows.append(
                {
                    "group": r["group"],
                    "score": r["normalized_damerau_levenshtein_distance"],
                    "target_position": start_pos + i,
                    "predicted_base": p,
                    "target_base": t,
                    "status": status,
                    "intact_mass": intact_mass,
                    "min_window_time": min_window_time,
                    "max_window_time": max_window_time,
                    "adduct_type": adduct_type,
                }
            )

    for b in range(len(target_sequence)):
        targ_base = masses.filter(pl.col("id") == target_sequence[b][0])[
            "encoding"
        ].item()
        match_rows.append(
            {
                "group": -1,
                "score": 0,
                "target_position": b,
                "predicted_base": targ_base,
                "target_base": targ_base,
                "status": "match",
                "intact_mass": np.nan,
                "min_window_time": np.nan,
                "max_window_time": np.nan,
                "adduct_type": "",
            }
        )

    df_expanded_alignment = pl.DataFrame(match_rows)

    df_expanded_alignment = df_expanded_alignment.sort(["group", "target_position"])

    df_expanded_alignment = df_expanded_alignment.with_columns(
        [
            pl.col("status").shift(-1).over("group").alias("next_status"),
            pl.col("predicted_base").shift(-1).over("group").alias("next_pred"),
            pl.col("target_base").shift(-1).over("group").alias("next_target"),
        ]
    )

    df_expanded_alignment = df_expanded_alignment.with_columns(
        (
            (pl.col("status") == "mismatch")
            & (pl.col("next_status") == "mismatch")
            & (pl.col("predicted_base") == pl.col("next_target"))
            & (pl.col("next_pred") == pl.col("target_base"))
        ).alias("is_swap_start")
    )

    df_expanded_alignment = df_expanded_alignment.with_columns(
        (
            pl.col("is_swap_start") | pl.col("is_swap_start").shift(1).over("group")
        ).alias("is_swap")
    )

    df_expanded_alignment = df_expanded_alignment.with_columns(
        pl.when(pl.col("is_swap"))
        .then(pl.lit("swap"))
        .otherwise(pl.col("status"))
        .alias("status")
    )

    df_expanded_alignment = df_expanded_alignment.drop(
        ["next_status", "next_pred", "next_target", "is_swap_start", "is_swap"]
    )

    df_expanded_alignment = df_expanded_alignment.join(
        masses.select(["encoding", "canonical_name"]),
        left_on="predicted_base",
        right_on="encoding",
        how="left",
    )

    df_expanded_alignment = df_expanded_alignment.sort(
        ["score", "group", "target_position"]
    )
    group_to_pos = {
        group: i
        for i, group in enumerate(
            df_expanded_alignment["group"].unique(maintain_order=True).to_list()
        )
    }
    df_expanded_alignment = df_expanded_alignment.with_columns(
        pos=pl.col("group").replace(group_to_pos)
    )

    return df_expanded_alignment


def plot_interpreted_alignment(df_expanded_alignment):

    color_scale = alt.Scale(
        domain=["match", "mismatch", "swap"], range=["black", "red", "gold"]
    )

    chart = (
        alt.Chart(df_expanded_alignment)
        .mark_text(fontSize=14, font="monospace")
        .encode(
            x=alt.X(
                "target_position:Q",
                title="Base position",
                scale=alt.Scale(
                    domain=[
                        df_expanded_alignment["target_position"].min() - 1,
                        df_expanded_alignment["target_position"].max() + 1,
                    ],
                    padding=0,
                ),
            ),
            y=alt.Y("pos:N"),
            text="predicted_base:N",
            color=alt.Color("status:N", scale=color_scale),
            tooltip=[
                "group",
                "target_position",
                "score",
                "predicted_base",
                "target_base",
                "status",
                "canonical_name",
                "intact_mass",
                "min_window_time",
                "max_window_time",
                "adduct_type",
            ],
        )
        .properties(width=850, height=400)
    )

    return chart


# Align predicted sequences to reference sequence and plot
df_alignment = align_prediction_results(prediction_vals, target_sequence, 1)
df_expanded_alignment = interpret_alignment_results(df_alignment, target_sequence)
chart = plot_interpreted_alignment(df_expanded_alignment)
chart.show()
chart.save(f"output/{sample_name}/alignment_output.html")
