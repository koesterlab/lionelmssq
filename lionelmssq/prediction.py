from dataclasses import dataclass
from pathlib import Path
from typing import List, Self
from lionelmssq.common import Side
from lionelmssq.linear_program import LinearProgramInstance
from lionelmssq.mass_table import DynamicProgrammingTable
from lionelmssq.masses import UNIQUE_MASSES, EXPLANATION_MASSES, MATCHING_THRESHOLD
import polars as pl
from loguru import logger

from lionelmssq.skeleton_building import SkeletonBuilder, \
    calculate_diff_dp, calculate_diff_errors


@dataclass
class Prediction:
    sequence: List[str]
    fragments: pl.DataFrame

    @classmethod
    def from_files(cls, sequence_path: Path, fragments_path: Path) -> Self:
        with open(sequence_path) as f:
            head, seq = f.readlines()
            assert head.startswith(">")

        fragments = pl.read_csv(fragments_path, separator="\t")
        return Prediction(sequence=seq.strip(), fragments=fragments)


class Predictor:
    def __init__(
        self,
        fragments: pl.DataFrame,
        seq_len: int,
        dp_table: DynamicProgrammingTable,
        unique_masses: pl.DataFrame = UNIQUE_MASSES,
        explanation_masses: pl.DataFrame = EXPLANATION_MASSES,
        matching_threshold: float = MATCHING_THRESHOLD,
        mass_tag_start: float = 0.0,
        mass_tag_end: float = 0.0,
    ):
        self.fragments = (
            fragments.with_row_index(name="orig_index")
            .sort("observed_mass")
            .with_row_index(name="index")
        )
        print(len(self.fragments))

        # Sort the fragments in the order of single nucleosides, start fragments, end fragments,
        # start fragments AND end fragments, internal fragments and then by mass for each category!

        with pl.Config() as cfg:
            cfg.set_tbl_rows(-1)
            print(
                self.fragments.select(
                    pl.col("observed_mass"),
                    pl.col("is_start"),
                    pl.col("is_end"),
                    pl.col("single_nucleoside"),
                    pl.col("is_start_end"),
                    pl.col("is_internal"),
                    # pl.col("mass_explanations"),
                    pl.col("index"),
                    pl.col("orig_index"),
                )
            )

        self.seq_len = seq_len
        self.explanations = {}
        self.mass_diffs = dict()
        self.mass_diffs_errors = dict()
        self.singleton_masses = None
        self.unique_masses = unique_masses
        self.explanation_masses = explanation_masses
        self.matching_threshold = matching_threshold
        self.mass_tags = {Side.START: mass_tag_start, Side.END: mass_tag_end}
        self.fragments_side = dict()
        self.fragment_masses = dict()
        self.dp_table = dp_table


    def predict(
            self, solver_params: dict, modification_rate: float = 0.5
        ) -> Prediction:
        # Adapt individual modification rates to universal one
        self.dp_table.adapt_individual_modification_rates_by_universal_one(
            modification_rate
        )

        # TODO: get rid of the requirement to pass the length of the sequence
        #  and instead infer it from the fragments

        # Collect the fragments for the start and end side which also include the start_end fragments (entire sequences)
        self.fragments_side[Side.START] = self.fragments.filter(
            pl.col("is_start") | pl.col("is_start_end")
        )
        self.fragments_side[Side.END] = self.fragments.filter(
            pl.col("is_end") | pl.col("is_start_end")
        )

        # Collect the masses of these fragments (subtract the appropriate tag masses)
        self.fragment_masses[Side.START] = self._collect_fragment_side_masses(
            fragments=self.fragments,
            side=Side.START,
        )
        self.fragment_masses[Side.END] = self._collect_fragment_side_masses(
            fragments=self.fragments,
            side=Side.END,
        )

        # Roughly estimate the differences as a first step with all fragments marked as start and then as end
        # Note that we do not consider fragments is_start_end now,
        # since the difference may be quite large and explained by lots of combinations
        # Note that there may be faulty mass fragments which will lead to bad (not truly existent) differences here!
        self.mass_diffs[Side.START] = self._collect_diffs(Side.START)
        self.mass_diffs[Side.END] = self._collect_diffs(Side.END)
        self.mass_diffs_errors[Side.START] = self._collect_diff_errors(Side.START)
        self.mass_diffs_errors[Side.END] = self._collect_diff_errors(Side.END)
        self.explanations = self._collect_diff_explanations(
            mass_diffs=self.mass_diffs,
            mass_diff_errors=self.mass_diffs_errors,
            modification_rate=modification_rate,
        )

        # TODO: Also consider that the observations are not complete and that
        #  we probably don't see all the letters as diffs or singletons.
        #  Hence, maybe do the following: Solve first with the reduced
        #  alphabet, and if the optimization does not yield a sufficiently
        #  good result, then try again with an extended alphabet.
        masses = (  # self.unique_masses
            self._reduce_alphabet()
        )

        skeleton_builder = SkeletonBuilder(
            fragment_masses=self.fragment_masses,
            fragments_side=self.fragments_side,
            mass_diffs=self.mass_diffs,
            explanations=self.explanations,
            seq_len=self.seq_len,
            matching_threshold=self.matching_threshold,
            mass_tags=self.mass_tags,
            dp_table=self.dp_table,
        )

        # Now we build the skeleton sequence from both sides and align them to get the final skeleton sequence!
        (
            skeleton_seq,
            start_fragments,
            end_fragments,
            invalid_start_fragments,
            invalid_end_fragments,
        ) = skeleton_builder.build_skeleton(modification_rate)

        # TODO: If the tags are considered in the LP at the end, then most of the following code will become obsolete!

        # We now create reduced self.fragments_side and their masses
        # which keeps the ordering of accepted start and end candidates while rejecting
        # the invalid ones, but keeping the ones with internal marking as internal candidates!
        self.fragments_side[Side.START] = self.fragments_side[Side.START].filter(
            ~pl.col("index").is_in(invalid_start_fragments)
        )

        # We also remove the fragments from the end side which are also present in the start side:
        self.fragments_side[Side.END] = (
            self.fragments_side[Side.END]
            .filter(~pl.col("index").is_in(invalid_end_fragments))
            .filter(~pl.col("index").is_in([i.index for i in start_fragments]))
        )

        # Collect the masses of the fragments for the _reduced_ start and end side
        self.fragment_masses[Side.START] = self._collect_fragment_side_masses(
            fragments=self.fragments_side[Side.START],
            side=Side.START,
        )
        self.fragment_masses[Side.END] = self._collect_fragment_side_masses(
            fragments=self.fragments_side[Side.END],
            side=Side.END,
        )

        def select_ends(fragments, idx):
            selected_fragments = [frag for frag in fragments if frag.index == idx]
            if len(selected_fragments) == 0:
                return 0, -1
            return selected_fragments[0].min_end, selected_fragments[0].max_end

        # Add new information from skeleton building to dataframe
        self.fragments = self.fragments.with_columns(
            (pl.col("index").is_in([frag.index for frag in start_fragments])).alias(
                "true_start"
            ),
            (pl.col("index").is_in([frag.index for frag in end_fragments])).alias(
                "true_end"
            ),
            (
                pl.col("is_internal")
                & ~pl.col("index").is_in(
                    [frag.index for frag in start_fragments + end_fragments]
                )
            ).alias("true_internal"),
            (
                pl.col("index").map_elements(
                    lambda x: select_ends(start_fragments + end_fragments, x)[0],
                    return_dtype=int,
                )
            ).alias("min_end"),
            (
                pl.col("index").map_elements(
                    lambda x: select_ends(start_fragments + end_fragments, x)[1],
                    return_dtype=int,
                )
            ).alias("max_end"),
        )

        # Rewriting the observed_mass column for the start and the end
        # fragments with the tag(s) subtracted masses for latter processing!
        self.fragments = pl.concat(
            [
                self.fragments.filter(pl.col("true_start")).replace_column(
                    self.fragments.get_column_index("observed_mass"),
                    pl.Series("observed_mass", self.fragment_masses[Side.START]),
                ),
                self.fragments.filter(~pl.col("true_start")),
            ]
        )
        # TODO: Remove usage of self.fragment_side and other unnecessary
        #  class variables completely
        self.fragments = pl.concat(
            [
                self.fragments.filter(pl.col("true_end")).replace_column(
                    self.fragments.get_column_index("observed_mass"),
                    pl.Series("observed_mass", self.fragment_masses[Side.END]),
                ),
                self.fragments.filter(~pl.col("true_end")),
            ]
        )

        print(
            "Number of internal fragments before filtering: ",
            len(self.fragments.filter(pl.col("true_internal"))),
        )

        # Filter out all internal fragments that do not fit anywhere in skeleton
        is_valid_fragment = []
        for frag in self.fragments.filter(pl.col("true_internal")).rows():
            filter_instance = LinearProgramInstance(
                fragments=self.fragments.filter(
                    pl.col("index") == frag[self.fragments.get_column_index("index")]
                ),
                nucleosides=masses,
                dp_table=self.dp_table,
                skeleton_seq=skeleton_seq,
                modification_rate=modification_rate,
            )
            if filter_instance.check_feasibility(
                solver_params=solver_params,
                threshold=MATCHING_THRESHOLD
                * frag[self.fragments.get_column_index("observed_mass")],
            ):
                is_valid_fragment.append(frag[self.fragments.get_column_index("index")])
        self.fragments = self.fragments.with_columns(
            true_internal=pl.when(pl.col("index").is_in(is_valid_fragment))
            .then(pl.col("true_internal"))
            .otherwise(False)
        )

        print(
            "Number of internal fragments after filtering: ",
            len(self.fragments.filter(pl.col("true_internal"))),
        )

        self.fragments = (
            self.fragments.filter(pl.col("true_start"))
            .vstack(self.fragments.filter(pl.col("true_end")))
            .vstack(self.fragments.filter(pl.col("true_internal")))
        )

        print(
            "Fragments considered for fitting, n_fragments = ",
            len(self.fragments.get_column("observed_mass").to_list()),
        )

        if not start_fragments:
            logger.warning(
                "No start fragments provided, this will likely lead to suboptimal results."
            )

        if not end_fragments:
            logger.warning(
                "No end fragments provided, this will likely lead to suboptimal results."
            )

        lp_instance = LinearProgramInstance(
            fragments=self.fragments,
            nucleosides=masses,
            dp_table=self.dp_table,
            skeleton_seq=skeleton_seq,
            modification_rate=modification_rate,
        )

        seq, fragment_predictions = lp_instance.evaluate(solver_params)

        fragment_predictions = pl.concat(
            [fragment_predictions, self.fragments.select(pl.col("orig_index"))],
            how="horizontal",
        )

        # reorder fragment predictions so that they match the original order again
        fragment_predictions = fragment_predictions.sort("orig_index")

        return Prediction(
            sequence=seq,
            fragments=fragment_predictions,
        )


    def _collect_fragment_side_masses(
        self, fragments, side: Side
    ):
        """
        Collect the fragment masses for the given side (also including the
        start_end fragments, i.e. the entire sequence).
        """

        # Collect the (tag subtracted) masses of the fragments for the side
        side_fragments = self._get_terminal_fragments_without_tags(
            side=side, fragments=fragments
        )
        # Collect the (both tags subtracted) masses of the start_end fragments
        start_end_fragments = self._get_terminal_fragments_without_tags(
            side=None, fragments=fragments
        )

        return side_fragments + start_end_fragments

    def _get_terminal_fragments_without_tags(self, side: Side, fragments:
    pl.DataFrame):
        if side is None:
            return [
                i - self.mass_tags[Side.START] - self.mass_tags[Side.END]
                for i in fragments.filter(pl.col("is_start_end"))
                .get_column("observed_mass")
                .to_list()
            ]
        return [
            i - self.mass_tags[side]
            for i in fragments.filter(pl.col(f"is_{side}"))
            .get_column("observed_mass")
            .to_list()
        ]

    def _collect_diffs(self, side: Side) -> list:
        masses = self.fragment_masses[side]

        return [masses[0]] + [masses[i] - masses[i - 1] for i in range(1, len(masses))]

    def _collect_diff_errors(self, side: Side) -> list:
        masses = self.fragment_masses[side]

        return [calculate_diff_errors(
            self.mass_tags[side], masses[0] + self.mass_tags[side],
            self.matching_threshold,
            )] + [calculate_diff_errors(
                masses[i] + self.mass_tags[side],
                masses[i - 1] + self.mass_tags[side],
                self.matching_threshold,
            )
            for i in range(1, len(masses))
            ]


    def _collect_diff_explanations(
            self, mass_diffs, mass_diff_errors, modification_rate
    ) -> dict:
        # Collect singleton masses
        singleton_masses = set(self.fragments.filter(pl.col("single_nucleoside")).get_column(
                    "observed_mass"
                ))

        explanations = {}
        for diff, diff_error in zip(
                mass_diffs[Side.START] + mass_diffs[Side.END],
                mass_diff_errors[Side.START] + mass_diff_errors[Side.END],
        ):
            explanations[diff] = calculate_diff_dp(
                diff, diff_error, modification_rate, self.seq_len, self.dp_table
            )

        for diff in singleton_masses:
            explanations[diff] = calculate_diff_dp(
                diff, self.matching_threshold, modification_rate,
                self.seq_len, self.dp_table
            )

        return explanations
        # TODO: Can make it simpler here by rejecting diff which cannot be explained instead of doing it in the _predict_skeleton function!


    def _reduce_alphabet(self) -> pl.DataFrame:
        observed_nucleosides = {
            nuc
            for expls in self.explanations.values()
            for expl in expls
            for nuc in expl
        }
        reduced = self.unique_masses.filter(
            pl.col("nucleoside").is_in(observed_nucleosides)
        )
        print("Nucleosides considered for fitting after alphabet reduction:", reduced)

        self.dp_table.adapt_individual_modification_rates_by_alphabet_reduction(
            observed_nucleosides
        )

        return reduced
