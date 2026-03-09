from dataclasses import dataclass
from pathlib import Path
from typing import List, Self, Set, Tuple
import polars as pl
from loguru import logger

from spectrseqtools.common import (
    calculate_error_threshold,
    calculate_explanations,
    parse_nucleosides,
)
from spectrseqtools.linear_program import LinearProgramInstance
from spectrseqtools.mass_explanation import is_valid_mass
from spectrseqtools.mass_table import DynamicProgrammingTable
from spectrseqtools.masses import PHOSPHATE_LINK_MASS
from spectrseqtools.skeleton_building import SkeletonBuilder


@dataclass
class Prediction:
    sequence: List[str]
    fragments: pl.DataFrame

    @classmethod
    def from_files(cls, sequence_path: Path, fragments_path: Path) -> Self:
        with open(sequence_path) as f:
            # Read only lines pertaining sequence in short format (consisting
            # only of representatives without mass-silent alternatives)
            head, seq = f.readlines()[:2]
            assert head.startswith(">")

        fragments = pl.read_csv(fragments_path, separator="\t")
        return Prediction(sequence=parse_nucleosides(seq.strip()), fragments=fragments)

    @classmethod
    def default(cls) -> Self:
        return Prediction(
            sequence=[],
            fragments=pl.DataFrame(
                schema={
                    "left": pl.Int64,
                    "right": pl.Int64,
                    "observed_mass": pl.Float64,
                    "standard_unit_mass": pl.Float64,
                    "predicted_mass": pl.Float64,
                    "predicted_diff": pl.Float64,
                    "predicted_seq": pl.String,
                    "orig_index": pl.UInt32,
                }
            ),
        )


class Predictor:
    def __init__(
        self,
        dp_table: DynamicProgrammingTable,
        nucleotide_df: pl.DataFrame,
    ):
        self.nucleotide_df = nucleotide_df
        self.dp_table = dp_table

    def predict(
        self,
        fragments: pl.DataFrame,
        solver_params: dict,
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

        fragments, explanations = self.filter_by_explanation(fragments)

        skeleton_builder = SkeletonBuilder(
            explanations=explanations,
            dp_table=self.dp_table,
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
        self.dp_table.print_masses()
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
            fragments = self.filter_with_lp(
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
                dp_table=self.dp_table,
                skeleton_seq=skeleton_seq,
            )

            return Prediction(*lp_instance.evaluate(solver_params))
        except Exception:
            return Prediction.default()

    def filter_by_explanation(
        self, fragments: pl.DataFrame
    ) -> Tuple[pl.DataFrame, dict]:
        old_alphabet_size = -1

        explanations = {}
        while old_alphabet_size != len(self.dp_table.masses):
            old_alphabet_size = len(self.dp_table.masses)
            # Roughly explain the mass differences (to reduce the alphabet)
            # Note there may be faulty mass fragments leading to not truly existent values
            explanations = self.collect_diff_explanations_for_su(fragments=fragments)

            # TODO: Also consider that the observations are not complete and that
            #  we probably don't see all the letters as diffs or singletons.
            #  Hence, maybe do the following: Solve first with the reduced
            #  alphabet, and if the optimization does not yield a sufficiently
            #  good result, then try again with an extended alphabet.

            # Reduce nucleotide alphabet based on fragments
            observed_nucleotides = {
                nuc
                for expls in explanations.values()
                if expls is not None
                for expl in expls
                for nuc in expl
            }
            fragments = self._reduce_alphabet(observed_nucleotides, fragments)

        print("Alphabet after explanation-based reduction:")
        self.dp_table.print_masses()
        print()

        return fragments, explanations

    def _reduce_alphabet(
        self, nucleotide_list: Set[str], fragments: pl.DataFrame
    ) -> pl.DataFrame:
        self.dp_table.adapt_individual_modification_rates_by_alphabet_reduction(
            nucleotide_list
        )

        # Filter out all fragments without any explanations
        return (
            fragments.with_columns(
                pl.struct("observed_mass", "standard_unit_mass")
                .map_elements(
                    lambda x: is_valid_mass(
                        mass=x["standard_unit_mass"],
                        dp_table=self.dp_table,
                        threshold=self.dp_table.tolerance * x["observed_mass"],
                    ),
                    return_dtype=bool,
                )
                .alias("is_valid")
            )
            .filter(pl.col("is_valid"))
            .drop("is_valid")
        )

    def filter_with_lp(
        self,
        fragments: pl.DataFrame,
        skeleton_seq: list,
        solver_params: dict,
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
                dp_table=self.dp_table,
                skeleton_seq=skeleton_seq,
            )

            # Check whether fragment can feasibly be aligned to skeleton
            if filter_instance.minimize_error(
                solver_params=solver_params
            ) > self.dp_table.tolerance * fragments.item(idx, "observed_mass"):
                is_invalid.append(fragments.item(idx, "index"))

        # Return only valid fragments
        return fragments.filter(~pl.col("index").is_in(is_invalid))

    def collect_diff_explanations_for_su(self, fragments: pl.DataFrame) -> dict:
        # Collect explanation for all reasonable mass differences for each side
        explanations = {
            **self.collect_explanations_per_side(
                fragments=fragments.filter(pl.col(
                    "fragmentation").str.contains("START")),
            ),
            **self.collect_explanations_per_side(
                fragments=fragments.filter(pl.col(
                    "fragmentation").str.contains("END")),
            ),
        }

        # Collect singleton masses
        singleton_list = fragments.filter(pl.col("is_singleton"))

        idx_observed_mass = fragments.get_column_index("observed_mass")
        idx_su_mass = fragments.get_column_index("standard_unit_mass")
        for singleton in singleton_list.rows():
            explanations[singleton[idx_su_mass]] = calculate_explanations(
                diff=singleton[idx_su_mass],
                threshold=self.dp_table.tolerance * singleton[idx_observed_mass],
                dp_table=self.dp_table,
            )

        return explanations

    def collect_explanations_per_side(self, fragments: pl.DataFrame) -> dict:
        max_weight = (
            max(self.nucleotide_df.get_column("nucleoside_mass").to_list())
            + PHOSPHATE_LINK_MASS
        )
        su_masses = fragments.get_column("standard_unit_mass").to_list()
        observed_masses = fragments.get_column("observed_mass").to_list()
        start = 0
        end = 1

        explanations = {}
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
                self.dp_table.tolerance,
            )
            expl = calculate_explanations(
                diff=diff,
                threshold=diff_error,
                dp_table=self.dp_table,
            )
            if expl is not None and len(expl) >= 1:
                explanations[diff] = expl
            if end == len(fragments) - 1:
                start += 1
            else:
                end += 1

        return explanations
