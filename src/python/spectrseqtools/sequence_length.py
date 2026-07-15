# -*- coding: utf-8 -*-
"""Module for sequence length estimation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Self, Set, Tuple

import numpy as np

from spectrseqtools.dataclasses import SolverParameters
from spectrseqtools.enums import LengthEstimatorMetric
from spectrseqtools.fragments import StandardUnitFragments
from spectrseqtools.prediction.composition_inference import CompositionInferrer
from spectrseqtools.prediction.sequence_inference import LinearProgramInstance
from spectrseqtools.sequence import SkeletonSequence


@dataclass
class SequenceLengthEstimator(ABC):
    """Class to estimate sequence length."""

    inferrer: CompositionInferrer

    @classmethod
    def with_metric(
        cls, metric: LengthEstimatorMetric, inferrer: CompositionInferrer, **kwargs
    ) -> Self:
        """Initialize subclass based on given metric."""
        match metric:
            case LengthEstimatorMetric.JACCARD:
                return JaccardIndexBasedEstimator(inferrer=inferrer)
            case LengthEstimatorMetric.LP:
                return LPBasedEstimator(inferrer=inferrer, **kwargs)
            case _:
                raise NotImplementedError(
                    f"Support for '{metric}' is currently not given."
                )

    @abstractmethod
    def select_sequence_length(
        self,
        start_skeleton: SkeletonSequence,
        end_skeleton: SkeletonSequence,
        start_fragments: StandardUnitFragments,
        end_fragments: StandardUnitFragments,
    ) -> int:
        """
        Select best sequence length.

        Parameters
        ----------
        start_skeleton : SkeletonSequence
            Skeleton in 5'-direction.
        end_skeleton : SkeletonSequence
            Skeleton in 3'-direction.
        start_fragments : StandardUnitFragments
            List of terminal fragments in 5'-direction.
        end_fragments : StandardUnitFragments
            List of terminal fragments in 3'-direction.

        Returns
        -------
        int
            Selected sequence length.

        """


@dataclass
class LPBasedEstimator(SequenceLengthEstimator):
    """Class to estimate sequence length based on LP score."""

    solver_params: SolverParameters

    def select_sequence_length(
        self,
        start_skeleton: SkeletonSequence,
        end_skeleton: SkeletonSequence,
        start_fragments: StandardUnitFragments,
        end_fragments: StandardUnitFragments,
    ) -> int:
        # Determine sequence length with the best LP score
        best_len = -1
        best_val = np.inf
        for len_cand in range(self.inferrer.seq.min_len, self.inferrer.seq.max_len + 1):
            # Merge directional skeletons
            seq = start_skeleton.merge(other=end_skeleton, seq_len=len_cand)

            # Skip candidates resulting in invalid sequences
            if not self.inferrer.seq.validate_sequence(
                seq=seq,
                alphabet=self.inferrer.alphabet,
            ):
                continue

            # Combine directional terminal-fragment lists
            fragments = StandardUnitFragments.from_terminals(
                start_fragments=start_fragments,
                end_fragments=end_fragments,
                seq_len=len_cand,
            )

            # Determine LP score for terminal-fragment alignment
            value = self._determine_lp_score(
                terminal_fragments=fragments,
                skeleton_seq=seq,
            )

            # Update best found sequence length if needed
            if value < best_val:
                best_val = value
                best_len = len_cand

        if best_len < 0:
            raise Exception(
                "No sequence length fitting the given sequence mass could be estimated."
            )

        return best_len

    def _determine_lp_score(
        self,
        terminal_fragments: StandardUnitFragments,
        skeleton_seq: SkeletonSequence,
    ) -> float:
        """

        Parameters
        ----------
        terminal_fragments : StandardUnitFragments
            Terminal SU-fragments.
        skeleton_seq : SkeletonSequence
            Skeleton sequence.

        Returns
        -------
        float
            Score of linear program solution.

        """
        # Initialize LP instance for terminal fragment
        try:
            lp_instance = LinearProgramInstance(
                fragments=terminal_fragments.fragments,
                alphabet=self.inferrer.alphabet,
                seq=self.inferrer.seq,
                skeleton_seq=skeleton_seq,
            )
        except Exception:
            return np.inf

        # Return minimum error when fragments can feasibly be aligned to skeleton
        return lp_instance.minimize_error(solver_params=self.solver_params)


class JaccardIndexBasedEstimator(SequenceLengthEstimator):
    """Class to estimate sequence length based on Jaccard index."""

    def select_sequence_length(
        self,
        start_skeleton: SkeletonSequence,
        end_skeleton: SkeletonSequence,
        start_fragments: StandardUnitFragments,
        end_fragments: StandardUnitFragments,
    ) -> int:
        # Determine sequence length with the highest similarity between skeleton parts
        best_len = self.inferrer.seq.min_len
        best_val = -1
        for len_cand in range(self.inferrer.seq.min_len, self.inferrer.seq.max_len + 1):
            # Skip candidates resulting in invalid sequences
            if not self.inferrer.seq.validate_sequence(
                seq=start_skeleton.merge(other=end_skeleton, seq_len=len_cand),
                alphabet=self.inferrer.alphabet,
            ):
                continue

            # Determine normalized sum of Jaccard similarity in each position
            value = (
                sum(
                    map(
                        self._determine_jaccard_score,
                        zip(
                            start_skeleton.sequence[:len_cand],
                            end_skeleton.sequence[len(end_skeleton) - len_cand :],
                        ),
                    )
                )
                / len_cand
            )

            # Update best found sequence length if needed
            if value > best_val:
                best_val = value
                best_len = len_cand

        if best_val < 0:
            raise Exception(
                "No sequence length fitting the given sequence mass could be estimated."
            )

        return best_len

    @staticmethod
    def _determine_jaccard_score(input_tuple: Tuple[Set[str], Set[str]]) -> float:
        """
        Determine Jaccard index.

        Parameters
        ----------
        input_tuple : Tuple[Set[str], Set[str]]
            Tuple of candidates for nucleotides at any skeleton position.

        Returns
        -------
        float
            Jaccard index.

        """
        # Return score for perfect similarity if one set is empty
        if len(input_tuple[0]) == 0 or len(input_tuple[1]) == 0:
            return 1

        # Return score for perfect similarity if one set is subset of the other
        if input_tuple[0].issuperset(input_tuple[1]) or input_tuple[1].issuperset(
            input_tuple[0]
        ):
            return 1

        # Return Jaccard score
        return len(input_tuple[0].intersection(input_tuple[1])) / len(
            input_tuple[0].union(input_tuple[1])
        )
