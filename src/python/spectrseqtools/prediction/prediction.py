# -*- coding: utf-8 -*-
"""Prediction of sequence and fragments."""

from typing import Set, Tuple

import yaml

from spectrseqtools.common import set_output_path
from spectrseqtools.dataclasses import Prediction, SequenceInformation, SolverParameters
from spectrseqtools.enums import SolverType
from spectrseqtools.fragments import RawFragments, StandardUnitFragments
from spectrseqtools.nucleotide_alphabet import NucleotideAlphabet
from spectrseqtools.parsers import PredictionOptions
from spectrseqtools.prediction.composition_inference import CompositionInferrer
from spectrseqtools.prediction.fragment_classification import FragmentClassifier
from spectrseqtools.prediction.sequence_inference import LinearProgramInstance
from spectrseqtools.prediction.skeleton_building import SkeletonBuilder

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


class Predictor:
    """Class to predict sequence and fragment."""

    def __init__(self, options: PredictionOptions):
        # Set parameters for LP solver
        self.solver_params = SolverParameters(
            solver=select_solver(options.solver),
            threads=options.threads,
            msg=False,
            time_limit_short=options.lp_timeout_short,
            time_limit_long=options.lp_timeout_long,
        )

        self.fragment_dir, self.file_prefix = set_output_path(
            input_path=options.fragments, output_dir=options.output_dir
        )

        # Initialize fragment classifier
        self.classifier = FragmentClassifier(file_path=options.meta)

        with open(options.meta, "r", encoding="utf-8") as f:
            meta = yaml.safe_load(f)

        self.intensity_cutoff = meta.setdefault(
            "intensity_cutoff", DEFAULT_INTENSITY_CUTOFF
        )

        # Initialize nucleotide alphabet
        alphabet = NucleotideAlphabet.from_file(
            modification_rate=options.modification_rate
        )
        max_weight = alphabet.max.nucleotide_mass
        alphabet.filter_by_singletons(singleton_path=options.singletons)

        # Standardize intact sequence mass by removing START_END fragmentation to gain SU mass
        seq_mass_obs = meta["intact_mass"]
        seq_mass_su = (
            seq_mass_obs - self.classifier.start_end_fragmentation * alphabet.precision
        )

        # Initialize SequenceInformation class
        seq_info = SequenceInformation(
            max_len=int(seq_mass_su / alphabet.min.nucleotide_mass),
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

        self.inferrer = inferrer
        self.max_weight = max_weight
        self.fragment_path = options.fragments
        self.predict_path = options.fragment_predictions
        self.sequence_path = options.sequence_prediction
        self.sequence_name = options.sequence_name

    def predict(self):
        """Predict sequence."""
        print("Alphabet after singleton reduction:")
        self.inferrer.print_alphabet()
        print()

        # Initialize raw fragments
        fragments = RawFragments.from_file(input_path=self.fragment_path)
        fragments.filter_by_intensity(cutoff=self.intensity_cutoff)

        # Classify raw fragments into SU-fragments
        fragments = self.classifier.classify(fragments=fragments)

        fragments.filter_by_intact_mass(seq_info=self.inferrer.seq)
        fragments.filter_with_traceback_matrix(inferrer=self.inferrer)

        # Save SU-fragments
        fragments.save(
            output_path=self.fragment_dir
            / f"{self.file_prefix}.standard_unit_fragments.tsv"
        )

        fragments.index()

        print("Number of fragments before prediction:", len(fragments))
        print()

        # Predict sequence
        prediction = self.predict_sequence(
            fragments=fragments,
            solver_params=self.solver_params,
        )

        print("Predicted sequence =\t", prediction.sequence)

        # Save prediction results
        prediction.save(
            fragment_path=self.predict_path,
            sequence_path=self.sequence_path,
            sequence_name=self.sequence_name,
            alphabet=self.inferrer.alphabet,
        )

        return prediction

    def predict_sequence(
        self,
        fragments: StandardUnitFragments,
        solver_params: SolverParameters,
    ) -> Prediction:
        """
        Predict sequence from fragments (with all modifications).

        Parameters
        ----------
        fragments : StandardUnitFragments
            SU-fragments for prediction.
        solver_params : SolverParameters
            Solver parameter.

        Returns
        -------
        Prediction
            Predicted sequence and fragments.

        """
        fragments, compositions = self.filter_by_composition(fragments)

        skeleton_builder = SkeletonBuilder(
            compositions=compositions,
            inferrer=self.inferrer,
        )

        # Build skeleton sequence from both sides and align them into final sequence
        try:
            skeleton_seq, fragments = skeleton_builder.build_skeleton(
                fragments=fragments, solver_params=solver_params
            )
        except Exception:
            return Prediction.default()

        print()
        print("Number of fragments before skeleton-based reduction:", len(fragments))

        # Reduce nucleotide alphabet based on skeleton
        fragments, mapping = self._reduce_alphabet(
            nucleotide_list=skeleton_seq.nucleotides, fragments=fragments
        )
        skeleton_seq.update_indexing(mapping=mapping)

        print("Number of fragments after skeleton-based reduction:", len(fragments))
        print()

        print("Alphabet after skeleton-based reduction:")
        self.inferrer.print_alphabet()
        print()

        # Filter out all internal fragments that do not fit anywhere in skeleton
        print("Number of internal fragments before filter: ", len(fragments.internal))

        # TODO: Investigate LP initialization of highly modified sequences
        #  for being wrong-length predictions (min/max length issue?)
        try:
            fragments.filter_with_linear_optimization(
                inferrer=self.inferrer,
                skeleton_seq=skeleton_seq,
                solver_params=solver_params,
            )
        except ValueError:
            return Prediction.default()

        print("Number of internal fragments after filter: ", len(fragments.internal))

        print()
        print("Number of fragments considered for fitting:", len(fragments))
        print()

        fragments.print_warning()

        # Remove ambiguities in skeleton by solving LP instance
        try:
            lp_instance = LinearProgramInstance(
                fragments=fragments.fragments,
                alphabet=self.inferrer.alphabet,
                seq=self.inferrer.seq,
                skeleton_seq=skeleton_seq,
            )

            return lp_instance.evaluate(solver_params=solver_params)
        except Exception:
            return Prediction.default()

    def filter_by_composition(
        self, fragments: StandardUnitFragments
    ) -> Tuple[StandardUnitFragments, dict]:
        """
        Filter nucleotide alphabet and fragments by composition validity.

        Parameters
        ----------
        fragments : StandardUnitFragments
            SU-fragments before reduction.

        Returns
        -------
        StandardUnitFragments
            SU-fragments before reduction.
        dict
            Dictionary of masses and their corresponding compositions.

        """
        old_alphabet_size = -1

        singleton_compositions = fragments.collect_singleton_compositions(
            inferrer=self.inferrer
        )
        while old_alphabet_size != len(self.inferrer.alphabet):
            old_alphabet_size = len(self.inferrer.alphabet)

            # Roughly infer compositions for mass differences (to reduce the alphabet)
            # Note there may be faulty mass fragments leading to not truly existent values
            compositions = {
                **singleton_compositions,
                **fragments.start.collect_mass_difference_compositions(
                    inferrer=self.inferrer,
                    max_weight=self.max_weight,
                ),
                **fragments.end.collect_mass_difference_compositions(
                    inferrer=self.inferrer,
                    max_weight=self.max_weight,
                ),
            }

            # TODO: Also consider that the observations are not complete and that
            #  we probably don't see all the letters as diffs or singletons.
            #  Hence, maybe do the following: Solve first with the reduced
            #  alphabet, and if the optimization does not yield a sufficiently
            #  good result, then try again with an extended alphabet.

            # Reduce nucleotide alphabet based on fragments
            observed_nucleotides = {
                nuc for comp in compositions.values() for nuc in comp.nucleotides
            }
            fragments, _ = self._reduce_alphabet(observed_nucleotides, fragments)

        print("Alphabet after composition-based reduction:")
        self.inferrer.print_alphabet()
        print()

        return fragments, compositions

    def _reduce_alphabet(
        self, nucleotide_list: Set[str], fragments: StandardUnitFragments
    ) -> Tuple[StandardUnitFragments, dict]:
        """
        Reduce nucleotide alphabet (and fragments) by list of valid nucleotides.

        Parameters
        ----------
        nucleotide_list : Set[str]
            List of valid nucleotides.
        fragments : StandardUnitFragments
            SU-fragments before reduction.

        Returns
        -------
        fragments : StandardUnitFragments
            SU-fragments after reduction.
        mapping : dict
            Mapping between old and new indexing.

        """
        # Reduce nucleotide alphabet
        mapping = (
            self.inferrer.adapt_individual_modification_rates_by_alphabet_reduction(
                nucleotide_list
            )
        )

        # Filter out all fragments with no valid composition
        fragments.filter_with_traceback_matrix(inferrer=self.inferrer)

        return fragments, mapping


def select_solver(solver: SolverType):
    """Select solver."""
    match solver:
        case SolverType.GUROBI:
            return "GUROBI_CMD"
        case SolverType.CBC:
            return "PULP_CBC_CMD"
        case _:
            raise NotImplementedError(f"Support for '{solver}' is currently not given.")
