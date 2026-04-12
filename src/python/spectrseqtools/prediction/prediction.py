from typing import Set, Tuple

import numpy as np
import polars as pl
from loguru import logger

from spectrseqtools.common import (
    calculate_compositions,
    calculate_error_threshold,
)
from spectrseqtools.dataclasses import Prediction, SolverParameters
from spectrseqtools.prediction.composition_inference import is_valid_mass
from spectrseqtools.prediction.sequence_inference import LinearProgramInstance
from spectrseqtools.prediction.skeleton_building import SkeletonBuilder
from spectrseqtools.prediction.traceback_matrix import CompositionInferrer


class Predictor:
    def __init__(
        self,
        inferrer: CompositionInferrer,
        nucleotide_df: pl.DataFrame,
    ):
        self.nucleotide_df = nucleotide_df
        self.inferrer = inferrer

    def predict(
        self,
        fragments: pl.DataFrame,
        solver_params: SolverParameters,
    ) -> Prediction:
        fragments = (
            fragments.with_row_index(name="orig_index")
            .sort("standard_unit_mass")
            .with_row_index(name="index")
        )
        print("Number of fragments before prediction:", len(fragments))
        print()

        fragments = fragments.with_columns(
            pl.lit(0, dtype=pl.Int64).alias("min_end"),
            pl.lit(-1, dtype=pl.Int64).alias("max_end"),
        )

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
        nucleotides = {nuc for skeleton_pos in skeleton_seq for nuc in skeleton_pos}
        fragments = self._reduce_alphabet(
            nucleotide_list=nucleotides, fragments=fragments
        )

        print("Number of fragments after skeleton-based reduction:", len(fragments))
        print()

        print("Alphabet after skeleton-based reduction:")
        self.inferrer.print_alphabet()
        print()

        # Filter out all internal fragments that do not fit anywhere in skeleton
        print(
            "Number of internal fragments before filtering: ",
            len(
                fragments.filter(
                    ~pl.col("fragmentation").str.contains("START")
                    & ~pl.col("fragmentation").str.contains("END")
                )
            ),
        )

        # TODO: Investigate LP initialization of highly modified sequences
        #  for being wrong-length predictions (min/max length issue?)
        try:
            fragments = self.filter_with_linear_optimization(
                fragments=fragments,
                skeleton_seq=skeleton_seq,
                solver_params=solver_params,
            )
        except ValueError:
            return Prediction.default()

        print(
            "Number of internal fragments after filtering: ",
            len(
                fragments.filter(
                    ~pl.col("fragmentation").str.contains("START")
                    & ~pl.col("fragmentation").str.contains("END")
                )
            ),
        )

        print()
        print("Number of fragments considered for fitting:", len(fragments))
        print()

        if len(fragments.filter(pl.col("fragmentation").str.contains("START"))) == 0:
            logger.warning(
                "No start fragments provided, this will likely lead to suboptimal results."
            )

        if len(fragments.filter(pl.col("fragmentation").str.contains("END"))) == 0:
            logger.warning(
                "No end fragments provided, this will likely lead to suboptimal results."
            )

        # Remove ambiguities in skeleton by solving LP instance
        try:
            lp_instance = LinearProgramInstance(
                fragments=fragments,
                inferrer=self.inferrer,
                skeleton_seq=skeleton_seq,
            )

            return lp_instance.evaluate(solver_params=solver_params)
        except Exception:
            return Prediction.default()

    def filter_by_composition(
        self, fragments: pl.DataFrame
    ) -> Tuple[pl.DataFrame, dict]:
        old_alphabet_size = -1

        compositions = {}
        while old_alphabet_size != len(self.inferrer.alphabet):
            old_alphabet_size = len(self.inferrer.alphabet)
            # Roughly infer compositions for mass differences (to reduce the alphabet)
            # Note there may be faulty mass fragments leading to not truly existent values
            compositions = self.collect_diff_compositions(fragments=fragments)

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
        self, nucleotide_list: Set[str], fragments: pl.DataFrame
    ) -> pl.DataFrame:
        self.inferrer.adapt_individual_modification_rates_by_alphabet_reduction(
            nucleotide_list
        )

        # Filter out all fragments with no valid composition
        return (
            fragments.with_columns(
                pl.struct("observed_mass", "standard_unit_mass")
                .map_elements(
                    lambda x: is_valid_mass(
                        mass=x["standard_unit_mass"],
                        inferrer=self.inferrer,
                        threshold=self.inferrer.tolerance * x["observed_mass"],
                    ),
                    return_dtype=bool,
                )
                .alias("is_valid")
            )
            .filter(pl.col("is_valid"))
            .drop("is_valid")
        )

    def filter_with_linear_optimization(
        self,
        fragments: pl.DataFrame,
        skeleton_seq: list,
        solver_params: SolverParameters,
    ) -> pl.DataFrame:
        is_invalid = []
        for idx in range(len(fragments)):
            # TODO: Add terminal-fragment filter based on LP output of
            #  sequence-length estimation and reuse the below (for speed-up)
            # # Skip terminal (i.e. non-internal) fragments
            # if ("START" in fragments.item(idx, "fragmentation")) or (
            #     "END" in fragments.item(idx, "fragmentation")
            # ):
            #     continue

            # Initialize LP instance for a singular fragment
            filter_instance = LinearProgramInstance(
                fragments=fragments[idx],
                inferrer=self.inferrer,
                skeleton_seq=skeleton_seq,
            )

            # Check whether fragment can feasibly be aligned to skeleton
            if filter_instance.minimize_error(
                solver_params=solver_params
            ) > self.inferrer.tolerance * fragments.item(idx, "observed_mass"):
                is_invalid.append(fragments.item(idx, "index"))

        # Return only valid fragments
        return fragments.filter(~pl.col("index").is_in(is_invalid))

    def collect_diff_compositions(self, fragments: pl.DataFrame) -> dict:
        # Collect compositions for all reasonable mass differences for each side
        compositions = {
            **self.collect_diff_compositions_per_side(
                fragments=fragments.filter(
                    pl.col("fragmentation").str.contains("START")
                ),
            ),
            **self.collect_diff_compositions_per_side(
                fragments=fragments.filter(pl.col("fragmentation").str.contains("END")),
            ),
        }

        # Determine all fragments that may be singletons
        fragments = fragments.with_columns(
            pl.struct("observed_mass", "standard_unit_mass")
            .map_elements(
                lambda x: is_singleton(
                    mass=x["standard_unit_mass"],
                    inferrer=self.inferrer,
                    threshold=self.inferrer.tolerance * x["observed_mass"],
                ),
                return_dtype=bool,
            )
            .alias("is_singleton")
        )
        # Collect singleton masses
        singleton_list = fragments.filter(pl.col("is_singleton"))

        idx_observed_mass = fragments.get_column_index("observed_mass")
        idx_su_mass = fragments.get_column_index("standard_unit_mass")
        for singleton in singleton_list.rows():
            compositions[singleton[idx_su_mass]] = calculate_compositions(
                diff=singleton[idx_su_mass],
                threshold=self.inferrer.tolerance * singleton[idx_observed_mass],
                inferrer=self.inferrer,
            )

        return compositions

    def collect_diff_compositions_per_side(self, fragments: pl.DataFrame) -> dict:
        max_weight = max(self.nucleotide_df.get_column("nucleotide_mass").to_list())
        su_masses = fragments.get_column("standard_unit_mass").to_list()
        observed_masses = fragments.get_column("observed_mass").to_list()
        start = 0
        end = 1

        compositions = {}
        while end < len(fragments):
            # Skip singletons
            if (end - start) <= 0:
                end += 1
                continue

            # Determine mass difference between fragments
            diff = su_masses[end] - su_masses[start]

            # If mass difference > any nucleotide mass, drop 1st fragment in window
            if diff > max_weight:
                start += 1
                end = start + 1
                continue

            diff_error = calculate_error_threshold(
                observed_masses[start],
                observed_masses[end],
                self.inferrer.tolerance,
            )
            comp = calculate_compositions(
                diff=diff,
                threshold=diff_error,
                inferrer=self.inferrer,
            )
            if comp is not None and len(comp) >= 1:
                compositions[diff] = comp
            if end == len(fragments) - 1:
                start += 1
            else:
                end += 1

        return compositions


def is_singleton(
    mass,
    inferrer: CompositionInferrer,
    threshold: float = None,
) -> bool:
    """
    Determine whether the given mass is associated with any singleton.

    Parameters
    ----------
    mass : float
        Given fragment mass.
    inferrer: CompositionInferrer
        Composition inferrer.
    threshold : float
        Error threshold.

    Returns
    -------
    bool
        Flag whether mass is associated with any singleton.

    """
    # Set singleton masses from alphabet
    singleton_masses = [mass.mass for mass in inferrer.alphabet]

    # Convert the target to an integer for easy operations
    target = int(round(mass / inferrer.precision, 0))

    # Set relative threshold if not given
    if threshold is None:
        threshold = inferrer.tolerance * mass

    # Convert the threshold to integer
    threshold = int(np.ceil(threshold / inferrer.precision))

    # Check whether a singleton mass could be found
    for value in range(target - threshold, target + threshold + 1):
        if value in singleton_masses:
            return True
    return False
