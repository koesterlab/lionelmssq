# -*- coding: utf-8 -*-
"""Prediction of sequence and fragments."""

from typing import Set, Tuple

from spectrseqtools.dataclasses import Prediction, SolverParameters
from spectrseqtools.fragments import StandardUnitFragments
from spectrseqtools.prediction.composition_inference import CompositionInferrer
from spectrseqtools.prediction.sequence_inference import LinearProgramInstance
from spectrseqtools.prediction.skeleton_building import SkeletonBuilder


class Predictor:
    """Class to predict sequence and fragment."""

    def __init__(
        self,
        inferrer: CompositionInferrer,
        max_weight: float,
    ):
        self.inferrer = inferrer
        self.max_weight = max_weight

    def predict(
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
        fragments = self._reduce_alphabet(
            nucleotide_list=skeleton_seq.nucleotides, fragments=fragments
        )

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
                inferrer=self.inferrer,
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
        while old_alphabet_size != self.inferrer.alphabet.size:
            old_alphabet_size = self.inferrer.alphabet.size

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
                nuc
                for comps in compositions.values()
                if comps is not None
                for comp in comps
                for nuc in comp
            }
            fragments = self._reduce_alphabet(observed_nucleotides, fragments)

        print("Alphabet after composition-based reduction:")
        self.inferrer.print_alphabet()
        print()

        return fragments, compositions

    def _reduce_alphabet(
        self, nucleotide_list: Set[str], fragments: StandardUnitFragments
    ) -> StandardUnitFragments:
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

        """
        # Reduce nucleotide alphabet
        self.inferrer.adapt_individual_modification_rates_by_alphabet_reduction(
            nucleotide_list
        )

        # Filter out all fragments with no valid composition
        fragments.filter_with_traceback_matrix(inferrer=self.inferrer)

        return fragments
