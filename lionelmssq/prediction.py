from dataclasses import dataclass
from itertools import chain, groupby
from pathlib import Path
from typing import List, Optional, Self, Set, Tuple
from lionelmssq.common import Side
from lionelmssq.linear_program import LinearProgramInstance
from lionelmssq.mass_table import DynamicProgrammingTable
from lionelmssq.masses import UNIQUE_MASSES, EXPLANATION_MASSES, MATCHING_THRESHOLD
from lionelmssq.mass_explanation import explain_mass
import polars as pl
from loguru import logger


@dataclass
class TerminalFragment:
    index: int  # fragment index
    min_end: int  # minimum length of fragment (negative for end fragments)
    max_end: int  # maximum length of fragment (negative for end fragments)


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
        solver: str,
        threads: int,
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
        self.solver = solver
        self.threads = threads
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

    def build_skeleton(
        self, modification_rate
    ) -> Tuple[
        List[Set[str]],
        List[TerminalFragment],
        List[TerminalFragment],
        List[int],
        List[int],
    ]:
        skeleton_seq_start, start_fragments, invalid_start_fragments = (
            self._predict_skeleton(Side.START, modification_rate)
        )

        print("Skeleton sequence start = ", skeleton_seq_start)

        skeleton_seq_end, end_fragments, invalid_end_fragments = self._predict_skeleton(
            Side.END, modification_rate
        )
        print("Skeleton sequence end = ", skeleton_seq_end)

        skeleton_seq = self._align_skeletons(skeleton_seq_start, skeleton_seq_end)

        print("Skeleton sequence = ", skeleton_seq)

        return (
            skeleton_seq,
            start_fragments,
            end_fragments,
            invalid_start_fragments,
            invalid_end_fragments,
        )

    def predict(self, modification_rate: float = 0.5) -> Prediction:
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
        self._collect_fragment_side_masses(Side.START)
        self._collect_fragment_side_masses(Side.END)

        # Collect the masses of the single nucleosides
        self._collect_singleton_masses()

        # Roughly estimate the differences as a first step with all fragments marked as start and then as end
        # Note that we do not consider fragments is_start_end now,
        # since the difference may be quite large and explained by lots of combinations
        # Note that there may be faulty mass fragments which will lead to bad (not truly existent) differences here!
        self._collect_diffs(Side.START)
        self._collect_diffs(Side.END)
        self._collect_diff_explanations(modification_rate)

        # TODO: Also consider that the observations are not complete and that
        #  we probably don't see all the letters as diffs or singletons.
        #  Hence, maybe do the following: Solve first with the reduced
        #  alphabet, and if the optimization does not yield a sufficiently
        #  good result, then try again with an extended alphabet.
        masses = (  # self.unique_masses
            self._reduce_alphabet()
        )

        # Now we build the skeleton sequence from both sides and align them to get the final skeleton sequence!
        (
            skeleton_seq,
            start_fragments,
            end_fragments,
            invalid_start_fragments,
            invalid_end_fragments,
        ) = self.build_skeleton(modification_rate)

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
        self._collect_fragment_side_masses(Side.START, restrict_is_start_end=True)
        self._collect_fragment_side_masses(Side.END, restrict_is_start_end=True)

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

        # TODO: Move solver selection in function called in init
        match self.solver:
            case "gurobi":
                solver_name = "GUROBI_CMD"
            case "cbc":
                solver_name = "PULP_CBC_CMD"
            case _:
                raise NotImplementedError(
                    f"Support for '{self.solver}' is currently not given."
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
                solver_name=solver_name,
                num_threads=self.threads,
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

        seq, fragment_predictions = lp_instance.evaluate(solver_name, self.threads)

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
        self, side: Side, restrict_is_start_end: bool = False
    ):
        # Collects the fragment masses for the given side (also includes the start_end fragments, i.e the entire sequence)
        # Optionally ``restrict_is_start_end`` can be set to True to only consider the start_end fragments which have been included in self.fragments_side[side]
        # This is useful later when a skeleton is already built and we need the masses of the accepted fragments!

        def _get_side_fragments(side: Side, fragments: pl.DataFrame):
            return [
                i - self.mass_tags[side]
                for i in fragments.filter(pl.col(f"is_{side}"))
                .get_column("observed_mass")
                .to_list()
            ]

        def _get_start_end_fragments(side: Side, fragments: pl.DataFrame):
            return [
                i - self.mass_tags[Side.START] - self.mass_tags[Side.END]
                for i in fragments.filter(pl.col("is_start_end"))
                .get_column("observed_mass")
                .to_list()
            ]

        if restrict_is_start_end:
            # Collect the (tag subtracted) masses of the fragments for the side
            side_fragments = _get_side_fragments(
                side=side, fragments=self.fragments_side[side]
            )
            # Collect the (both tags subtracted) masses of the start_end fragments
            start_end_fragments = _get_start_end_fragments(
                side=side, fragments=self.fragments_side[side]
            )
        else:
            # Collect the (tag subtracted) masses of the fragments for the side
            side_fragments = _get_side_fragments(side=side, fragments=self.fragments)
            # Collect the (both tags subtracted) masses of the start_end fragments
            start_end_fragments = _get_start_end_fragments(
                side=side, fragments=self.fragments
            )

        self.fragment_masses[side] = side_fragments + start_end_fragments

    def _align_skeletons(self, skeleton_seq_start, skeleton_seq_end) -> List[Set[str]]:
        # While building the ladder it may happen that things are unambiguous from one side, but not from the other!
        # In that case, we should consider the unambiguous side as the correct one! If the intersection is empty, then we can consider the union of the two!

        # Align the skeletons of the start and end fragments to get the final skeleton sequence!
        # Wherever there is no ambiguity, that nucleotide is preferrentially considered!

        skeleton_seq = [set() for _ in range(self.seq_len)]
        for i in range(self.seq_len):
            skeleton_seq[i] = skeleton_seq_start[i].intersection(skeleton_seq_end[i])
            if not skeleton_seq[i]:
                skeleton_seq[i] = skeleton_seq_start[i].union(skeleton_seq_end[i])

        # TODO: Its more complicated, since if two positions are ambiguous,
        #  they are not independent. If one nucleotide is selected this way,
        #  then the same nucleotide cannot be selected in the other position!

        return skeleton_seq

    def _collect_diffs(self, side: Side) -> None:
        masses = self.fragment_masses[side]

        self.mass_diffs[side] = [masses[0]] + [
            masses[i] - masses[i - 1] for i in range(1, len(masses))
        ]
        self.mass_diffs_errors[side] = [
            self._calculate_diff_errors(
                self.mass_tags[side],
                masses[0] + self.mass_tags[side],
                self.matching_threshold,
            )
        ] + [
            self._calculate_diff_errors(
                masses[i] + self.mass_tags[side],
                masses[i - 1] + self.mass_tags[side],
                self.matching_threshold,
            )
            for i in range(1, len(masses))
        ]

    def _calculate_diff_errors(self, mass1, mass2, threshold) -> float:
        retval = threshold * ((mass1**2 + mass2**2) ** 0.5) / abs(mass1 - mass2)
        # Constrain the maximum relative error to 1!
        # For mass difference very close to zero, the relative error can be very high!
        if retval > 1:
            retval = 1.0
        return retval

    def _collect_singleton_masses(
        self,
    ) -> None:
        masses = self.fragments.filter(pl.col("single_nucleoside")).get_column(
            "observed_mass"
        )
        self.singleton_masses = set(masses)

    def _calculate_diff_dp(self, diff, threshold, modification_rate):
        # TODO: Add support for other breakages than 'c/y_c/y'
        explanation_list = list(
            [
                entry
                for entry in explain_mass(
                    diff,
                    dp_table=self.dp_table,
                    seq_len=self.seq_len,
                    max_modifications=round(modification_rate * self.seq_len),
                    threshold=threshold,
                )
                if entry.breakage == "c/y_c/y"
            ][0].explanations
        )
        if len(explanation_list) > 0:
            retval = [
                Explanation(*explanation_list[i]) for i in range(len(explanation_list))
            ]
        else:
            retval = []
        return retval

    def _collect_diff_explanations(self, modification_rate) -> None:
        diffs = (self.mass_diffs[Side.START]) + (self.mass_diffs[Side.END])

        diffs_errors = (
            (self.mass_diffs_errors[Side.START]) + (self.mass_diffs_errors[Side.END])
        )
        for diff, diff_error in zip(diffs, diffs_errors):
            self.explanations[diff] = self._calculate_diff_dp(
                diff, diff_error, modification_rate
            )

        for diff in self.singleton_masses:
            self.explanations[diff] = self._calculate_diff_dp(
                diff, self.matching_threshold, modification_rate
            )
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

    def _predict_skeleton(
        self,
        side: Side,
        modification_rate,
        skeleton_seq: Optional[List[Set[str]]] = None,
        fragment_masses=None,
        candidate_fragments=None,
    ) -> Tuple[List[Set[str]], List[TerminalFragment], List[int]]:
        if skeleton_seq is None:
            skeleton_seq = [set() for _ in range(self.seq_len)]

        factor = 1 if side == Side.START else -1

        if not fragment_masses:
            fragment_masses = self.fragment_masses[side]

        if not candidate_fragments:
            candidate_fragments = (
                self.fragments_side[side].get_column("index").to_list()
            )

        def get_possible_nucleosides(pos: int, i: int) -> Set[str]:
            return skeleton_seq[pos + factor * i]

        def is_valid_pos(pos: int, ext: int) -> bool:
            pos = pos + factor * ext
            return (
                # 0 <= pos <= self.seq_len
                0 <= pos < self.seq_len
                if side == Side.START
                # else -(self.seq_len + 1) <= pos < 0
                else -self.seq_len <= pos < 0
            )

        pos = {0} if side == Side.START else {-1}

        # We reject some masses/some fragments which are not explained well by mass differences.
        # While iterating through the fragments, the "carry_over_mass" keeps a track of the rejected mass difference.
        # This is added to "diff" to get the next mass_difference.
        carry_over_mass = 0

        # "last_mass_valid" keeps a track of the last mass which was NOT rejected.
        # This is useful for calculating the difference threshold
        last_mass_valid = 0.0

        valid_terminal_fragments = []
        invalid_fragments = []
        for fragment_index, (diff, mass) in enumerate(
            zip(self.mass_diffs[side], fragment_masses)
        ):
            diff += carry_over_mass
            assert pos

            is_valid = True
            if diff in self.explanations:
                explanations = self.explanations.get(diff, [])
            else:
                threshold = self._calculate_diff_errors(
                    last_mass_valid + self.mass_tags[side],
                    last_mass_valid + diff + self.mass_tags[side],
                    self.matching_threshold,
                )
                explanations = self._calculate_diff_dp(
                    diff, threshold, modification_rate
                )

            # METHOD: if there is only one single nucleoside explanation, we can
            # directly assign the nucleoside if there are tuples, we have to assign a
            # possibility to the current and next position
            # and take care of expressing that both permutations are possible
            # Further, we consider multiple possible start positions, depending on
            # whether the previous explanations had different lengths.
            if explanations:
                next_pos = set()
                if side is Side.START:
                    min_p = min(pos)
                    max_p = max(pos)
                else:
                    min_p = max(pos)
                    max_p = min(pos)
                min_fragment_end = None
                max_fragment_end = None

                for p in pos:
                    p_specific_explanations = [
                        expl for expl in explanations if is_valid_pos(p, len(expl))
                    ]
                    alphabet_per_expl_len = {
                        expl_len: set(chain(*expls))
                        for expl_len, expls in groupby(p_specific_explanations, len)
                    }

                    if p_specific_explanations:
                        if p == min_p:
                            min_fragment_end = min_p + factor * min(
                                expl_len for expl_len in alphabet_per_expl_len
                            )
                        elif p == max_p:
                            max_fragment_end = max_p + factor * max(
                                expl_len for expl_len in alphabet_per_expl_len
                            )

                    # constrain already existing sets in the range of the expl
                    # by the nucs that are given in the explanations
                    for expl_len, alphabet in alphabet_per_expl_len.items():
                        for i in range(expl_len):
                            possible_nucleosides = get_possible_nucleosides(p, i)

                            if possible_nucleosides.issuperset(alphabet):
                                # If the current explanation sharpens the list of possibilities, clear all
                                # prior possibilities before the new explanation will add the sharpened ones below
                                possible_nucleosides.clear()

                            for j in alphabet:
                                possible_nucleosides.add(j)
                                # TODO: We need to do this better.
                                # Instead of adding just the letters, we somehow need to keep a track of the possibilities to be able to constrain the LP!
                                # We also then probably need part of the code immediately below!

                    # add possible follow up positions
                    next_pos.update(
                        p + factor * expl_len for expl_len in alphabet_per_expl_len
                    )
                if max_fragment_end is None:
                    max_fragment_end = min_fragment_end
                if min_fragment_end is None:
                    min_fragment_end = max_fragment_end
                if min_fragment_end is None:
                    # still None => both are None
                    # TODO can we stop early in building the ladder?
                    is_valid = False
            elif (
                abs(diff) <= self.matching_threshold * abs(mass + self.mass_tags[side])
                # Problem! The above approach might blow up if the masses are very close, i.e. diff is very close to zero!
            ):
                if side == Side.START:
                    min_fragment_end = min(pos)
                    max_fragment_end = max(pos)
                else:
                    min_fragment_end = max(pos)
                    max_fragment_end = min(pos)
                next_pos = pos
            else:
                is_valid = False

            if is_valid:
                valid_terminal_fragments.append(
                    TerminalFragment(
                        index=candidate_fragments[fragment_index],
                        min_end=min_fragment_end,
                        max_end=max_fragment_end,
                    )
                )
                pos = next_pos
                last_mass_valid = mass
                carry_over_mass = 0.0
            else:
                logger.warning(
                    f"Skipping {side} fragment {fragment_index} with observed mass {mass} because no "
                    "explanations are found for the mass difference."
                )
                carry_over_mass = diff

                # Consider the skipped fragments as internal fragments! Add back the terminal mass to this fragments!
                if fragment_index and candidate_fragments:
                    invalid_fragments.append(candidate_fragments[fragment_index])

        return skeleton_seq, valid_terminal_fragments, invalid_fragments


class Explanation:
    def __init__(self, *nucleosides):
        self.nucleosides = tuple(sorted(nucleosides))

    def __iter__(self):
        yield from self.nucleosides

    def __len__(self):
        return len(self.nucleosides)

    def __repr__(self):
        return f"{{{','.join(self.nucleosides)}}}"
