import yaml

from spectrseqtools.common import set_output_path
from spectrseqtools.dataclasses import SequenceInformation, SolverParameters
from spectrseqtools.enums import SolverType
from spectrseqtools.fragments import RawFragments
from spectrseqtools.nucleotide_alphabet import NucleotideAlphabet
from spectrseqtools.parsers import Options, PredictionOptions
from spectrseqtools.prediction.composition_inference import CompositionInferrer
from spectrseqtools.prediction.fragment_classification import FragmentClassifier
from spectrseqtools.prediction.prediction import Predictor
from spectrseqtools.preprocessing.preprocessing import Preprocessor

# Set default value for intensity cutoff
DEFAULT_INTENSITY_CUTOFF = 115000

# Set relative tolerance such that we consider
# abs(sum(masses)/target_mass - 1) < TOLERANCE for matching
# Note that the error is on the higher side than would be for a good
# calibrated machine (10 ppm), but in the absence of an experimental
# measurement of this error, this conservative value works well
TOLERANCE = 10e-6

# Set number of binary-compressed masses per integer cell in traceback matrix
COMPRESSION_RATE = 32


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

    # Initialize fragment classifier
    classifier = FragmentClassifier(file_path=options.meta)

    with open(options.meta, "r") as f:
        meta = yaml.safe_load(f)

    intensity_cutoff = meta.setdefault("intensity_cutoff", DEFAULT_INTENSITY_CUTOFF)

    # Initialize nucleotide alphabet
    alphabet = NucleotideAlphabet.from_file(modification_rate=options.modification_rate)
    max_weight = alphabet.max
    alphabet.filter_by_singletons(singleton_path=options.singletons)

    # Standardize intact sequence mass by removing START_END fragmentation to gain SU mass
    seq_mass_obs = meta["intact_mass"]
    seq_mass_su = seq_mass_obs - classifier.start_end_fragmentation * alphabet.precision

    # Initialize SequenceInformation class
    seq_info = SequenceInformation(
        max_len=int(seq_mass_su / alphabet.min),
        su_mass=seq_mass_su,
        obs_mass=seq_mass_obs,
        modification_rate=options.modification_rate,
    )

    # Initialize CompositionInferrer class
    inferrer = CompositionInferrer(
        alphabet=alphabet,
        compression_rate=int(COMPRESSION_RATE),
        tolerance=TOLERANCE,
        seq=seq_info,
    )

    print("Alphabet after singleton reduction:")
    inferrer.print_alphabet()
    print()

    # Initialize raw fragments
    fragments = RawFragments.from_file(input_path=options.fragments)
    fragments.filter_by_intensity(cutoff=intensity_cutoff)

    # Classify raw fragments into SU-fragments
    fragments = classifier.classify(fragments=fragments)

    fragments.filter_by_intact_mass(seq_info=seq_info)
    fragments.filter_with_traceback_matrix(inferrer=inferrer)

    # Save SU-fragments
    fragments.save(
        output_path=fragment_dir / f"{file_prefix}.standard_unit_fragments.tsv"
    )

    fragments.index()

    print("Number of fragments before prediction:", len(fragments))
    print()

    # Predict sequence
    prediction = Predictor(
        inferrer=inferrer,
        max_weight=max_weight,
    ).predict(
        fragments=fragments,
        solver_params=solver_params,
    )

    print("Predicted sequence =\t", prediction.sequence)

    # Save prediction results
    prediction.save(
        fragment_path=options.fragment_predictions,
        sequence_path=options.sequence_prediction,
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
