# -*- coding: utf-8 -*-
"""Prediction of sequence and fragments."""

from typing import Set, Tuple
import polars as pl
import yaml

from spectrseqtools.dataclasses import (
    FilterParameters,
    Prediction,
    PredictionFileSettings,
    SequenceInformation,
    SolverParameters,
)
from spectrseqtools.enums import SolverType
from spectrseqtools.error_calculator import ErrorCalculator
from spectrseqtools.fragments import RawFragments, StandardUnitFragments
from spectrseqtools.nucleotide_alphabet import NucleotideAlphabet
from spectrseqtools.parsers import PredictionOptions
from spectrseqtools.prediction.composition_inference import MatrixBasedInferrer
from spectrseqtools.prediction.fragment_classification import FragmentClassifier
from spectrseqtools.prediction.sequence_inference import LinearProgramInstance
from spectrseqtools.prediction.skeleton_building import SkeletonBuilder
from spectrseqtools.prediction.traceback_matrix import TracebackMatrix
from spectrseqtools.sequence_length import SequenceLengthEstimator


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

        # Set file-related settings
        self.file_settings = PredictionFileSettings(
            input_path=options.fragments,
            meta_path=options.meta,
            alphabet_path=options.alphabet,
            output_dir=options.output_dir,
            predicted_fragment_path=options.fragment_predictions,
            sequence_path=options.sequence_prediction,
            sequence_header=options.sequence_name,
        )

        # Initialize error calculator with desired metric
        error_calculator = ErrorCalculator.with_metric(
            metric=options.error_metric,
            tolerance=options.tolerance,
            decimal_places=options.num_decimal_places,
        )

        # Initialize fragment classifier
        self.classifier = FragmentClassifier(
            file_path=self.file_settings.meta_path,
            error=error_calculator,
            reduced=options.reduce_fragmentation_dict,
        )

        with open(self.file_settings.meta_path, "r", encoding="utf-8") as f:
            meta = yaml.safe_load(f)

        # Set intensity cutoff
        intensity_cutoff = None
        if "intensity_cutoff" in meta:
            intensity_cutoff = meta["intensity_cutoff"]

        # Set filter parameters
        self.filter_params = FilterParameters(
            intensity_cutoff=intensity_cutoff,
            cutoff_percentile=options.intensity_cutoff_percentile,
            nuc_weight_factor=options.composition_filter_weight_factor,
        )

        print("Intensity cutoff percentile:", self.filter_params.cutoff_percentile)

        # Initialize nucleotide alphabet
        if isinstance(self.file_settings.alphabet_path, pl.DataFrame):
            alphabet = NucleotideAlphabet.from_file(
                modification_rate=options.modification_rate,
                input_path=self.file_settings.alphabet_path,
                error=error_calculator,
            )
        else:
            alphabet = NucleotideAlphabet.from_dataframe(
                modification_rate=options.modification_rate,
                masses=self.file_settings.alphabet_path,
                error=error_calculator,
            )

        # Standardize intact sequence mass by removing START_END fragmentation to gain SU mass
        seq_mass_obs = meta["intact_mass"]
        seq_mass_su = round(
            seq_mass_obs - self.classifier.start_end_fragmentation,
            error_calculator.decimal_places,
        )

        # Initialize SequenceInformation class
        seq_info = SequenceInformation(
            max_len=int(seq_mass_su / alphabet.min.nucleotide_mass),
            su_mass=seq_mass_su,
            obs_mass=seq_mass_obs,
            modification_rate=options.modification_rate,
            max_variance=options.max_intact_mass_variance,
        )

        # Initialize CompositionInferrer class with traceback matrix
        matrix = TracebackMatrix.load_with_compression(
            alphabet=alphabet, compression_rate=int(options.compression_rate)
        )
        inferrer = MatrixBasedInferrer(
            alphabet=alphabet,
            error=error_calculator,
            matrix=matrix,
            seq=seq_info,
        )
        self.inferrer = inferrer

        self.estimator = SequenceLengthEstimator.with_metric(
            metric=options.length_estimator_metric,
            inferrer=self.inferrer,
            solver_params=self.solver_params,
        )

    def predict(self):
        """Predict sequence."""
        # TODO: Log to stderr instead of stdout
        print("Alphabet after singleton reduction:")
        self.inferrer.print_alphabet()
        print()

        # Initialize raw fragments
        fragments = RawFragments.from_file(
            input_path=self.file_settings.raw_fragment_path
        )
        fragments.filter_by_intensity(filter_params=self.filter_params)

        # Classify raw fragments into SU-fragments
        fragments = self.classifier.classify(fragments=fragments)

        fragments.filter_by_intact_mass(seq_info=self.inferrer.seq)
        fragments.filter_with_traceback_matrix(inferrer=self.inferrer)

        # Save SU-fragments
        fragments.save(output_path=self.file_settings.su_fragment_path)

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
            file_settings=self.file_settings,
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
        fragments, compositions = self.filter_by_composition(fragments=fragments)

        skeleton_builder = SkeletonBuilder(
            compositions=compositions,
            inferrer=self.inferrer,
        )

        # Build skeleton sequence from both sides and align them into final sequence
        try:
            skeleton_seq, fragments = skeleton_builder.build_skeleton(
                fragments=fragments,
                estimator=self.estimator,
            )
        # TODO: Replace generic Exception, within custom one
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
        # TODO: Replace generic ValueError, within custom one
        except ValueError or IndexError:
            # TODO: Replace IndexError for LP initialization with custom one
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
        # TODO: Replace generic Exception, within custom one
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

            max_weight = (
                self.inferrer.alphabet.max.nucleotide_mass
                * self.filter_params.nuc_weight_factor
            )

            # Roughly infer compositions for mass differences (to reduce the alphabet)
            # Note there may be faulty mass fragments leading to not truly existent values
            compositions = {
                **singleton_compositions,
                **fragments.start.collect_mass_difference_compositions(
                    inferrer=self.inferrer,
                    max_weight=max_weight,
                ),
                **fragments.end.collect_mass_difference_compositions(
                    inferrer=self.inferrer,
                    max_weight=max_weight,
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
            fragments, _ = self._reduce_alphabet(
                nucleotide_list=observed_nucleotides, fragments=fragments
            )

        print("Alphabet after composition-based reduction:")
        self.inferrer.print_alphabet()
        print()

        return fragments, compositions

    def _reduce_alphabet(
        self, nucleotide_list: Set[int], fragments: StandardUnitFragments
    ) -> Tuple[StandardUnitFragments, dict]:
        """
        Reduce nucleotide alphabet (and fragments) by list of valid nucleotides.

        Parameters
        ----------
        nucleotide_list : Set[int]
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
        mapping = self.inferrer.reduce_alphabet(new_alphabet=nucleotide_list)

        # Filter out all fragments with no valid composition
        fragments.filter_with_traceback_matrix(inferrer=self.inferrer)

        return fragments, mapping


def select_solver(solver: SolverType) -> str:
    """Select solver."""
    match solver:
        case SolverType.CBC:
            return "PULP_CBC_CMD"
        case SolverType.GUROBI:
            return "GUROBI_CMD"
        case SolverType.HIGHS:
            return "HiGHS_CMD"
        case _:
            raise NotImplementedError(f"Support for '{solver}' is currently not given.")
