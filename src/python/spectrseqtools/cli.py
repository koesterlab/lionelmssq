import os
import polars as pl
import yaml
import ddargparse
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Tuple

from spectrseqtools.common import set_output_path
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
from spectrseqtools.prediction.fragment_classification import classify_fragments
from spectrseqtools.prediction.prediction import Predictor
from spectrseqtools.prediction.traceback_matrix import (
    CompositionInferrer,
    SequenceInformation,
)
from spectrseqtools.preprocessing.preprocessing import AveragineBackbone, Preprocessor


class SolverType(Enum):
    CBC = "cbc"
    GUROBI = "gurobi"


@dataclass
class PreprocessingOptions(ddargparse.OptionsBase):
    """Preprocessing of raw data into fragments"""

    input: Path = field(
        metadata={"help": "Path to input file in RAW format"},
    )
    meta: Path = field(metadata={"help": "Path to YAML with meta information"})
    alphabet: Path | None = field(
        metadata={"help": "Path to file containing nucleotide alphabet."}
    )
    output_dir: Path | None = field(
        metadata={
            "help": "Output directory (default: input directory)",
        }
    )
    charge_range: Tuple[int, int] | None = field(
        metadata={
            "help": "Charge range considered for deconvolution "
            "(used in ms_deisotope package)."
        }
    )
    min_intensity: float | None = field(
        metadata={
            "help": "Minimum intensity required for peak consideration "
            "(used in ms_deisotope package as 'minimum_intensity')."
        }
    )
    tolerance: float = field(
        default=10e-6,
        metadata={"help": "Error tolerance to consider masses identical."},
    )
    boundary_factor: int = field(
        default=2,
        metadata={"help": "Factor for scaling theoretical singleton boundaries."},
    )
    min_precursor_charge: int = field(
        default=3,
        metadata={"help": "Minimum MS1 charge to consider associated MS2 scans."},
    )
    isotopic_shift_factor: int = field(
        default=10,
        metadata={"help": "Factor for scaling isotopic shift for precursors."},
    )
    envelope_min_score: float = field(
        default=150.0,
        metadata={
            "help": "Minimum accepted score during envelope fitting "
            "(used in ms_deisotope package as 'minimum_score')."
        },
    )
    envelope_error_tol: float = field(
        default=0.02,
        metadata={
            "help": "Error tolerance for envelopes during fitting "
            "(used in ms_deisotope package as 'mass_error_tolerance')."
        },
    )
    averagine_backbone: AveragineBackbone = field(
        default=AveragineBackbone.PHOSPHATE,
        metadata={
            "help": "Backbone considered in Averagine model "
            "(used in ms_deisotope package)."
        },
    )
    max_missed_peaks: int = field(
        default=1,
        metadata={
            "help": "Maximum number of missed peaks tolerated in envelope fitting "
            "(used in ms_deisotope package)."
        },
    )
    scale_method: str = field(
        default="sum",
        metadata={
            "help": "Scale method for intensity values (used in ms_deisotope package)."
        },
    )
    peak_error_tol: float = field(
        default=2e-5,
        metadata={
            "help": "Error tolerance for each individual peak "
            "(used in ms_deisotope package as 'error_tol')."
        },
    )
    truncate_after: float = field(
        default=0.9,
        metadata={
            "help": "Percentage of included isotopic patterns "
            "(used in ms_deisotope package)."
        },
    )
    cutoff_percentile: int = field(
        default=75, metadata={"help": "Intensity percentile used as cutoff"}
    )


@dataclass
class PredictionOptions(ddargparse.OptionsBase):
    """Prediction of sequence based on preprocessed fragments"""

    fragments: Path = field(
        metadata={"help": "Path to TSV table of observed fragments"},
    )
    meta: Path = field(metadata={"help": "Path to YAML with meta information"})
    singletons: Path = field(
        metadata={"help": "Path to TSV with singleton information"}
    )
    fragment_predictions: Path = field(
        metadata={
            "help": "Path to TSV table that shall contain the per fragment predictions"
        }
    )
    sequence_prediction: Path = field(
        metadata={
            "help": "Path to FASTA file that shall contain the predicted sequence"
        }
    )
    sequence_name: str = field(metadata={"help": "Header in FASTA output file"})
    output_dir: Path | None = field(
        metadata={
            "help": "Output directory (default: input directory)",
        }
    )
    modification_rate: float = field(
        default=0.5,
        metadata={
            "help": "Maximum percentage of modification in sequence",
        },
    )
    solver: SolverType = field(
        default=SolverType.GUROBI,
        metadata={"help": "Solver to use for optimization problem"},
    )
    lp_timeout_short: int = field(
        default=5, metadata={"help": "Time-out for shorter solving of LP instances"}
    )
    lp_timeout_long: int = field(
        default=60, metadata={"help": "Time-out for longer solving of LP instances"}
    )
    threads: int = field(
        default=1,
        metadata={"help": "Number of threads to use for the optimization problem"},
    )


@dataclass
class Options(ddargparse.OptionsBase):
    """
    De novo prediction of RNA sequences

    Usage:

    1. Preprocess raw data to gain fragments for prediction (grouped into
    files based on sequence).
    2. Predict sequence individually for each file output by preprocessing.

    """

    preprocessing: PreprocessingOptions | None
    prediction: PredictionOptions | None


def main():
    options = Options.parse_args()

    # Preprocess raw data
    if options.preprocessing is not None:
        Preprocessor(options=options.preprocessing).preprocess()

    # Predict sequence
    if options.prediction is not None:
        _ = predict(options=options.prediction)


def predict(options):
    # Set parameters for LP solver
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
