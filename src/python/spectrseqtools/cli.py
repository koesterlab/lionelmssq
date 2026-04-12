import polars as pl
import yaml

from spectrseqtools.common import set_output_path
from spectrseqtools.dataclasses import SolverParameters
from spectrseqtools.enums import SolverType
from spectrseqtools.masses import (
    COMPRESSION_RATE,
    DEFAULT_INTENSITY_CUTOFF,
    PRECISION,
    TOLERANCE,
    build_fragmentation_dict,
)
from spectrseqtools.nucleotide_alphabet import NucleotideAlphabet
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
    # Set parameters for LP solver
    solver_params = SolverParameters(
        solver=select_solver(options.solver),
        threads=options.threads,
        msg=False,
        time_limit_short=options.lp_timeout_short,
        time_limit_long=options.lp_timeout_long,
    )

    fragment_dir, file_prefix = set_output_path(
        input_path=options.fragments, output_dir=options.output_dir
    )

    with open(options.meta, "r") as f:
        meta = yaml.safe_load(f)

    # Read preprocessed fragments
    fragments = pl.read_csv(options.fragments, separator="\t")

    # Initialize nucleotide alphabet
    alphabet = NucleotideAlphabet.from_file()
    alphabet.filter_by_singleton_selection(singleton_path=options.singletons)

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
        max_len=int(seq_mass_su / alphabet.min_mass()),
        su_mass=seq_mass_su,
        obs_mass=seq_mass_obs,
        modification_rate=options.modification_rate,
    )

    # Initialize CompositionInferrer class
    inferrer = CompositionInferrer(
        nucleotide_df=alphabet.nucleotides,
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
        nucleotide_df=alphabet.nucleotides,
    ).predict(
        fragments=fragments,
        solver_params=solver_params,
    )

    print("Predicted sequence =\t", prediction.sequence)

    # Save fragment predictions
    prediction.fragments.save(output_path=options.fragment_predictions)

    # Save predicted sequence
    prediction.sequence.save(
        output_path=options.sequence_prediction,
        sequence_name=options.sequence_name,
        alphabet=alphabet,
    )

    return prediction


def select_solver(solver: SolverType):
    match solver:
        case SolverType.GUROBI:
            return "GUROBI_CMD"
        case SolverType.CBC:
            return "PULP_CBC_CMD"
        case _:
            raise NotImplementedError(f"Support for '{solver}' is currently not given.")
