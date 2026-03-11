import os
import polars as pl
import yaml
from pathlib import Path
from tap import Tap
from typing import List, Literal

from spectrseqtools.fragment_classification import classify_fragments
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
from spectrseqtools.prediction import Predictor
from spectrseqtools.preprocessing import preprocess
from spectrseqtools.traceback_matrix import CompositionInferrer, SequenceInformation


class Settings(Tap):
    fragments: Path  # Path to TSV table or RAW data of observed fragments to use for prediction
    meta: Path  # Path to YAML with meta information to use for prediction
    fragment_predictions: (
        Path  # Path to TSV table that shall contain the per fragment predictions
    )
    sequence_prediction: (
        Path  # Path to FASTA file that shall contain the predicted sequence
    )
    output_dir: Path = None  # Output directory (default: input directory)
    sequence_name: str
    modification_rate: float = 0.5  # Maximum percentage of modification in sequence
    solver: Literal["gurobi", "cbc"] = (
        "gurobi"  # Solver to use for the optimization problem
    )
    lp_timeout_short: int = 5  # Time-out for shorter solving of LP instances
    lp_timeout_long: int = 60  # Time-out for longer solving of LP instances
    cutoff_percentile: int = 75  # Intensity percentile used as cutoff
    threads: int = 1  # Number of threads to use for the optimization problem


def main():
    settings = Settings(underscores_to_dashes=True).parse_args()

    # Set parameters for LP solver
    solver_params = {
        "fixed": {
            "solver": select_solver(settings.solver),
            "threads": settings.threads,
            "msg": False,
        },
        "timeLimit(short)": settings.lp_timeout_short,
        "timeLimit(long)": settings.lp_timeout_long,
    }

    settings.fragments = settings.fragments.resolve()
    fragment_dir = (
        settings.fragments.parent
        if settings.output_dir is None
        else settings.output_dir
    )
    file_prefix = settings.fragments.stem
    with open(settings.meta, "r") as f:
        meta = yaml.safe_load(f)

    # Preprocess data if necessary
    match settings.fragments.suffix:
        case ".raw":
            print("RAW file found. Preprocessing raw data...")
            # Preprocess raw data
            fragments, singletons, meta = preprocess(
                file_path=settings.fragments,
                deconvolution_params={},
                meta_params=meta,
                cutoff_percentile=settings.cutoff_percentile,
            )
            # Save preprocessed fragments
            fragments.write_csv(fragment_dir / f"{file_prefix}.tsv", separator="\t")

            # Save singletons detected from raw data
            singletons.write_csv(
                fragment_dir / f"{file_prefix}.singletons.tsv", separator="\t"
            )

            # Save updated meta data
            with open(fragment_dir / f"{file_prefix}.preprocessed.meta.yaml", "w") as f:
                yaml.dump(meta, f)

            print("Preprocessing completed!\n")
        case ".tsv":
            print("TSV file found. Proceeding without preprocessing.")
            # Read already preprocessed fragments
            fragments = pl.read_csv(settings.fragments, separator="\t")

            # Read singletons if given
            singletons = None
            if os.path.isfile(fragment_dir / f"{file_prefix}.singletons.tsv"):
                singletons = pl.read_csv(
                    fragment_dir / f"{file_prefix}.singletons.tsv", separator="\t"
                )
        case _:
            raise NotImplementedError(
                "Support is currently only given for TSV or RAW files."
            )

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
    start_tag = meta.setdefault("label_mass_5T", 555.1294)
    end_tag = meta.setdefault("label_mass_3T", 455.1491)

    # Build fragmentation dict
    fragmentation_dict = build_fragmentation_dict(
        mass_5_prime=start_tag, mass_3_prime=end_tag
    )

    # Standardize sequence mass (remove START_END fragmentation to gain SU mass)
    seq_mass_obs = meta["sequence_mass"]
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
        modification_rate=settings.modification_rate,
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
    prediction.fragments.write_csv(settings.fragment_predictions, separator="\t")

    # Save predicted sequence
    with open(settings.sequence_prediction, "w") as f:
        print(f">{settings.sequence_name}", file=f)
        print("".join(prediction.sequence), file=f)
        print(f">{settings.sequence_name}_full", file=f)
        print(format_sequence_to_full_version(seq=prediction.sequence), file=f)


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


def select_solver(solver: str):
    match solver:
        case "gurobi":
            return "GUROBI_CMD"
        case "cbc":
            return "PULP_CBC_CMD"
        case _:
            raise NotImplementedError(f"Support for '{solver}' is currently not given.")
