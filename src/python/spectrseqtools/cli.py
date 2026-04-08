import os
import polars as pl
import yaml
from typing import List

from spectrseqtools.common import set_output_path
from spectrseqtools.enums import SolverType
from spectrseqtools.masses import (
    COMPRESSION_RATE,
    DEFAULT_INTENSITY_CUTOFF,
    NUC_REPS,
    NUCLEOTIDE_DF,
    PRECISION,
    TOLERANCE,
    UNMODIFIED_BASES,
    build_fragmentation_dict,
)
from spectrseqtools.parsers import Options, PredictionOptions
from spectrseqtools.prediction.fragment_classification import classify_fragments
from spectrseqtools.prediction.prediction import Predictor
from spectrseqtools.prediction.traceback_matrix import (
    CompositionInferrer,
    SequenceInformation,
)
from spectrseqtools.preprocessing.preprocessing import Preprocessor


def main():
    options = Options.parse_args()

    # Preprocess raw data
    if options.preprocessing is not None:
        Preprocessor(options=options.preprocessing).preprocess()

    # Predict sequence
    if options.prediction is not None:
        _ = predict(options=options.prediction)


def predict(options: PredictionOptions):
    solver_params = {
        "fixed": {
            "solver": select_solver(options.solver),
            "threads": options.threads,
            "msg": False,
        },
        "timeLimit(short)": options.lp_timeout_short,
        "timeLimit(long)": options.lp_timeout_long,
    }

    fragment_dir, file_prefix = set_output_path(
        input_path=options.fragments, output_dir=options.output_dir
    )

    with open(options.meta, "r") as f:
        meta = yaml.safe_load(f)

    # Read preprocessed fragments
    fragments = pl.read_csv(options.fragments, separator="\t")

    # Read singletons if given
    singletons = None
    if os.path.isfile(options.singletons):
        singletons = pl.read_csv(options.singletons, separator="\t")

    print("Singletons identified during preprocessing:", singletons)
    print()

    nucleotide_df = NUCLEOTIDE_DF

    # Filter by singletons
    if singletons is not None:
        # Map singletons to their mass representative
        singletons = singletons.with_columns(
            pl.col("id").replace_strict(NUC_REPS).alias("id")
        )

        # Select only bases found in singletons
        nucleotide_df = nucleotide_df.with_columns(
            pl.when(
                pl.col("representative").is_in(singletons.get_column("id").to_list())
            )
            .then(pl.col("modification_rate"))
            .otherwise(pl.lit(0.0))
            .alias("modification_rate")
        )

    # Ensure modification rates of unmodified bases are set to 1
    nucleotide_df = nucleotide_df.with_columns(
        pl.when(~pl.col("representative").is_in(UNMODIFIED_BASES))
        .then(pl.col("modification_rate"))
        .otherwise(pl.lit(1.0))
        .alias("modification_rate")
    )

    # Read additional parameter from meta file
    intensity_cutoff = meta.setdefault("intensity_cutoff", DEFAULT_INTENSITY_CUTOFF)
    start_tag = meta.setdefault("5_prime_tag", 555.1294)
    end_tag = meta.setdefault("3_prime_tag", 455.1491)

    # Build fragmentation dict
    fragmentation_dict = build_fragmentation_dict(start_tag=start_tag, end_tag=end_tag)

    # Standardize intact sequence mass by removing START_END fragmentation to
    # gain SU mass
    seq_mass_obs = meta["intact_mass"]
    seq_mass_su = (
        seq_mass_obs
        - [
            mass * PRECISION
            for mass in fragmentation_dict
            if "START_END" in fragmentation_dict[mass]
        ][0]
    )

    # Initialize SequenceInformation class
    seq_info = SequenceInformation(
        max_len=int(
            seq_mass_su
            / PRECISION
            / min(
                pl.Series(
                    nucleotide_df.filter(pl.col("modification_rate") > 0.0).select(
                        "integer_mass"
                    )
                ).to_list()
            )
        ),
        su_mass=seq_mass_su,
        obs_mass=seq_mass_obs,
        modification_rate=options.modification_rate,
    )

    # Initialize CompositionInferrer class
    inferrer = CompositionInferrer(
        nucleotide_df=nucleotide_df,
        compression_rate=int(COMPRESSION_RATE),
        tolerance=TOLERANCE,
        precision=PRECISION,
        seq=seq_info,
    )

    print("Alphabet after singleton reduction:")
    inferrer.print_alphabet()
    print()

    # Classify preprocessed fragments
    fragments = classify_fragments(
        fragment_masses=fragments,
        inferrer=inferrer,
        fragmentation_dict=fragmentation_dict,
        output_file_path=fragment_dir / f"{file_prefix}.standard_unit_fragments.tsv",
        intensity_cutoff=intensity_cutoff,
    )

    # Predict sequence
    prediction = Predictor(
        inferrer=inferrer,
        nucleotide_df=nucleotide_df,
    ).predict(
        fragments=fragments,
        solver_params=solver_params,
    )

    print("Predicted sequence =\t", prediction.sequence)

    # Save fragment predictions
    prediction.fragments.write_csv(options.fragment_predictions, separator="\t")

    # Save predicted sequence
    with open(options.sequence_prediction, "w") as f:
        print(f">{options.sequence_name}", file=f)
        print("".join(prediction.sequence), file=f)
        print(f">{options.sequence_name}_full", file=f)
        print(format_sequence_to_full_version(seq=prediction.sequence), file=f)

    return prediction


def format_sequence_to_full_version(seq: List[str]) -> str:
    """
    Format a sequence to its full version (i.e. include alternate nucleotides).

    Parameters
    ----------
    seq: List[str]
        Given predicted sequence.

    Returns
    -------
    str
        Sequence with all alternate nucleotides.

    """
    output = ""
    for nuc in seq:
        alt_nucs = (
            NUCLEOTIDE_DF.filter(pl.col("representative") == nuc)
            .select("id_list")
            .item()
            .to_list()
        )
        if len(alt_nucs) == 1:
            output += nuc
        else:
            output += "[" + "|".join(alt_nucs) + "]"
    return output


def select_solver(solver: SolverType):
    match solver:
        case SolverType.GUROBI:
            return "GUROBI_CMD"
        case SolverType.CBC:
            return "PULP_CBC_CMD"
        case _:
            raise NotImplementedError(f"Support for '{solver}' is currently not given.")
