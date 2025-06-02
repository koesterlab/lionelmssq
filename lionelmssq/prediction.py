from dataclasses import dataclass
from itertools import chain, combinations, groupby
from pathlib import Path
from typing import List, Optional, Self, Set, Tuple
from pulp import (
    LpProblem,
    LpMinimize,
    LpInteger,
    LpContinuous,
    LpVariable,
    lpSum,
    getSolver,
)
from lionelmssq.common import (
    Side,
    get_singleton_set_item,
    milp_is_one,
    TerminalFragment,
    Explanation,
)
from lionelmssq.graph_skeleton import construct_graph_skeleton
from lionelmssq.alignment import (
    align_skeletons,
    align_list_explanations,
    align_skeletons_multi_seq,
)
from lionelmssq.masses import UNIQUE_MASSES, EXPLANATION_MASSES, MATCHING_THRESHOLD
from lionelmssq.mass_explanation import explain_mass
import polars as pl
from loguru import logger
from copy import deepcopy
from math import e

LP_relaxation_threshold = 0.9


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
        fragments: pl.DataFrame = None,
        seq_len: int = 10,
        solver: str = "cbc",
        threads: int = 1,
        unique_masses: pl.DataFrame = UNIQUE_MASSES,
        explanation_masses: pl.DataFrame = EXPLANATION_MASSES,
        matching_threshold: float = MATCHING_THRESHOLD,
        mass_tag_start: float = 0.0,
        mass_tag_end: float = 0.0,
        print_mass_table: bool = False,
    ):
        if fragments is not None:
            self.fragments = (
                fragments.with_row_index(name="orig_index")
                .sort("observed_mass")
                .with_row_index(name="index")
            )

        # Sort the fragments in the order of single nucleosides, start fragments, end fragments,
        # start fragments AND end fragments, internal fragments and then by mass for each category!

        if print_mass_table:
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

        # Defining functions on the classes defined in other files:
        self._graph_skeleton = lambda *args, **kwargs: construct_graph_skeleton(
            self, *args, **kwargs
        )
        self._align_skeletons = lambda *args, **kwargs: align_skeletons(
            self, *args, **kwargs
        )
        self._align_list_explanations = lambda *args, **kwargs: align_list_explanations(
            self, *args, **kwargs
        )
        self._align_skeletons_multi_seq = (
            lambda *args, **kwargs: align_skeletons_multi_seq(self, *args, **kwargs)
        )

    def predict(
        self, num_top_paths=100, consider_variable_sequence_lengths=True
    ) -> Prediction:

        nucleosides, nucleoside_masses = self._calculate_diffs_and_nucleosides()

        #Reduces the possible nucelotide space:
        self.explanation_masses = self.explanation_masses.filter(
            pl.col("nucleoside").is_in(nucleosides)
        )

        # Now we build the skeleton sequence from both sides and align them to get the final skeleton sequence!
        (
            _,
            start_fragments,
            skeleton_seq_start,
            invalid_start_fragments,
            seq_score_start,
            list_explanations_start,
        ) = self._graph_skeleton(
            Side.START,
            use_ms_intensity_as_weight=False,
            num_top_paths=num_top_paths,
            peanlize_explanation_length_params={"zero_len_weight": 0.0, "base": e},
            consider_variable_sequence_lengths=consider_variable_sequence_lengths,
        )

        print("Fitting end fragments now")

        (
            _,
            end_fragments,
            skeleton_seq_end,
            invalid_end_fragments,
            seq_score_end,
            list_explanations_end,
        ) = self._graph_skeleton(
            Side.END,
            use_ms_intensity_as_weight=False,
            num_top_paths=num_top_paths,
            peanlize_explanation_length_params={"zero_len_weight": 0.0, "base": e},
            consider_variable_sequence_lengths=consider_variable_sequence_lengths,
        )

        print("Aligning sequences now")

        seq_set, list_set, start_seq_index, end_seq_index = (
            self._align_list_explanations(
                list_explanations_start, list_explanations_end
            )
        )

        # TODO: Write a function to remove the 'duplicates' from the seq_set!
        # They are not all literally duplicates, but effectively will be duplicates!
        # Some may actually be literally duplicates!

        if not seq_set:
            logger.warning(
                "No perfect start-end list alignment found, resorting to best possible sequence alignment ranked using scores."
            )
            seq_set, score, start_seq_index, end_seq_index = (
                self._align_skeletons_multi_seq(
                    skeleton_seq_start=skeleton_seq_start,
                    skeleton_seq_end=skeleton_seq_end,
                    score_seq_start=seq_score_start,
                    score_seq_end=seq_score_end,
                    nucleosides=nucleosides,
                )
            )

            # TODO: Check this function execution here if the start and end lengths don't match!

            # Print the indices of seq_set corresponding to the top score
            top_score = max(score)
            top_score_indices = [idx for idx, sc in enumerate(score) if sc == top_score]
            print("Top sequences = ", [seq_set[i] for i in top_score_indices])
            print("Top sequences scores = ", top_score)
            print("Other_sequences = ", seq_set)

            chosen_seq_index = top_score_indices[0]
        else:
            chosen_seq_index = 0

        # Choose a skeleton sequence from the seq_set for further optimization
        # since the optimizer can only handle a single sequence in terms of sets at the time!
        skeleton_seq = seq_set[chosen_seq_index]
        start_fragments = start_fragments[start_seq_index[chosen_seq_index]]
        end_fragments = end_fragments[end_seq_index[chosen_seq_index]]
        invalid_start_fragments = invalid_start_fragments[
            start_seq_index[chosen_seq_index]
        ]
        invalid_end_fragments = invalid_end_fragments[end_seq_index[chosen_seq_index]]

        # Update sequence length with the chosen list explanation:
        self.seq_len = len(list(chain.from_iterable(skeleton_seq)))

        # print("Multi-Graph aligned_skeleton_seq selected = ", skeleton_seq)
        if list_set:
            print("Top 100 Multi-Graph aligned_skeleton_LIST selected:")
            for item in list_set[:100]:
                print(item, " with length = ", len(list(chain.from_iterable(item))))
        else:
            print("Multi-Graph aligned_skeleton_seq selected = ", skeleton_seq)

        # Remove the start and end fragments which are the same, this is not expected!
        # Consider them only with start fragments!
        for start in start_fragments:
            for end in end_fragments:
                if start.index == end.index:
                    end_fragments.remove(end)

        # We now create reduced self.fragments_side and their masses
        # which keeps the ordereing of accepted start and end candidates while rejecting
        # the invalid ones, but keeping the ones with internal marking as internal candidates!

        # TODO: If the tags are considered in the LP at the end, then most of the following code will become obsolete!
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

        # Rewriting the observed_mass column for the start and the end fragment_sides
        # with the tag(s) subtracted masses for latter processing!
        self.fragments_side[Side.START].replace_column(
            self.fragments_side[Side.START].get_column_index("observed_mass"),
            pl.Series("observed_mass", self.fragment_masses[Side.START]),
        )
        self.fragments_side[Side.END].replace_column(
            self.fragments_side[Side.END].get_column_index("observed_mass"),
            pl.Series("observed_mass", self.fragment_masses[Side.END]),
        )

        # Create a data frame for the internal fragments:
        self.fragments_internal = (
            self.fragments.filter(pl.col("is_internal")).filter(
                ~pl.col("index").is_in(
                    [i.index for i in start_fragments]
                    + [i.index for i in end_fragments]
                )
            )
            # .filter(pl.col("intensity") > 50000)
        )  # TODO: REMOVE THIS INTENSITY FILTER!! OR ADD AS NEEDED!

        # TODO: One can recheck the explanations for the internal fragments, if they match with the ladder sequence.
        # Remove the ones that do not!! Easily implemenented!

        print("Number of internal fragments = ", len(self.fragments_internal))

        self.fragments = (
            self.fragments_side[Side.START]
            .vstack(self.fragments_side[Side.END])
            .vstack(self.fragments_internal)
        )

        fragment_masses = (
            self.fragment_masses[Side.START]
            + self.fragment_masses[Side.END]
            + [i for i in self.fragments_internal.get_column("observed_mass").to_list()]
        )

        prob = LpProblem("RNA sequencing", LpMinimize)
        # i = 1,...,n: positions in the sequence
        # j = 1,...,m: fragments
        # b = 1,...,k: (modified) bases

        fragment_masses = self.fragments.get_column("observed_mass").to_list()
        n_fragments = len(fragment_masses)
        print("Fragments considered for fitting, n_fragments = ", n_fragments)

        valid_fragment_range = list(range(n_fragments))

        prob = LpProblem("RNA sequencing", LpMinimize)
        # i = 1,...,n: positions in the sequence
        # j = 1,...,m: fragments
        # b = 1,...,k: (modified) bases

        if not start_fragments:
            logger.warning(
                "No start fragments provided, this will likely lead to suboptimal results."
            )

        if not end_fragments:
            logger.warning(
                "No end fragments provided, this will likely lead to suboptimal results."
            )

        # TODO: IMP: Initiate the variables with the nucleotides (but don't fix them) that are already known from the dp!

        # x: binary variables indicating fragment j presence at position i
        x = [
            [
                LpVariable(f"x_{i},{j}", lowBound=0, upBound=1, cat=LpInteger)
                for j in valid_fragment_range
            ]
            for i in range(self.seq_len)
        ]
        # y: binary variables indicating base b at position i
        y = [
            {
                b: LpVariable(f"y_{i},{b}", lowBound=0, upBound=1, cat=LpInteger)
                for b in nucleosides
            }
            for i in range(self.seq_len)
        ]
        # z: binary variables indicating product of x and y
        z = [
            [
                {
                    b: LpVariable(
                        f"z_{i},{j},{b}", lowBound=0, upBound=1, cat=LpInteger
                    )
                    for b in nucleosides
                }
                for j in valid_fragment_range
            ]
            for i in range(self.seq_len)
        ]
        # weight_diff_abs: absolute value of weight_diff
        predicted_mass_diff_abs = [
            LpVariable(f"predicted_mass_diff_abs_{j}", lowBound=0, cat=LpContinuous)
            for j in valid_fragment_range
        ]
        # weight_diff: difference between fragment monoisotopic mass and sum of masses of bases in fragment as estimated in the MILP
        predicted_mass_diff = [
            fragment_masses[j]
            - lpSum(
                [
                    z[i][j][b] * nucleoside_masses[b]
                    for i in range(self.seq_len)
                    for b in nucleosides
                ]
            )
            for j in valid_fragment_range
        ]

        # optimization function
        prob += lpSum([predicted_mass_diff_abs[j] for j in valid_fragment_range])

        # select one base per position
        for i in range(self.seq_len):
            prob += lpSum([y[i][b] for b in nucleosides]) == 1

        # fill z with the product of binary variables x and y
        for i in range(self.seq_len):
            for j in valid_fragment_range:
                for b in nucleosides:
                    prob += z[i][j][b] <= x[i][j]
                    prob += z[i][j][b] <= y[i][b]
                    prob += z[i][j][b] >= x[i][j] + y[i][b] - 1

        # ensure that fragment is aligned continuously
        # (no gaps: if x[i1,j] = 1 and x[i2,j] = 1, then x[i_between,j] = 1)
        for j in valid_fragment_range:
            for i1, i2 in combinations(range(self.seq_len), 2):
                # i2 and i1 are inclusive
                assert i2 > i1
                if i2 - i1 > 1:
                    prob += (x[i1][j] + x[i2][j] - 1) * (i2 - i1 - 1) <= lpSum(
                        [x[i_between][j] for i_between in range(i1 + 1, i2)]
                    )

        # ensure that start fragments are aligned at the beginning of the sequence
        for fragment in start_fragments:
            # j is the row index where the "index" matches fragment.index
            # fragment.index uses the original (mass sorted) index of the read fragment files,
            # but in self.fragments we disqualify many fragments of the original file.
            # Hence, we need to find the correct row index in self.fragments which corresponds to the original index
            # since in the MILP we fit all the fragments of self.fragments
            j = (
                self.fragments.with_row_index("row_index")
                .filter(pl.col("index") == fragment.index)
                .item(0, "row_index")
            )
            # min_end is exclusive
            for i in range(fragment.min_end):
                x[i][j].setInitialValue(1)
                x[i][j].fixValue()
            for i in range(fragment.max_end, self.seq_len):
                x[i][j].setInitialValue(0)
                x[i][j].fixValue()

        # ensure that end fragments are aligned at the end of the sequence
        for fragment in end_fragments:
            # j is the row index where the "index" matches fragment.index
            j = (
                self.fragments.with_row_index("row_index")
                .filter(pl.col("index") == fragment.index)
                .item(0, "row_index")
            )
            # min_end is exclusive
            for i in range(fragment.min_end + 1, 0):
                x[i][j].setInitialValue(1)
                x[i][j].fixValue()
            for i in range(-self.seq_len, fragment.max_end + 1):
                x[i][j].setInitialValue(0)
                x[i][j].fixValue()

        # Fragments that aren't either start or end are either inner or uncertain.
        # Hence, we don't further constrain their positioning and length and let the
        # LP decide.

        # constrain weight_diff_abs to be the absolute value of weight_diff
        for j in valid_fragment_range:
            # if j not in invalid_start_fragments and j not in invalid_end_fragments:
            prob += predicted_mass_diff_abs[j] >= predicted_mass_diff[j]
            prob += predicted_mass_diff_abs[j] >= -predicted_mass_diff[j]

        # use skeleton seq to fix bases
        for i, nucs in enumerate(skeleton_seq):
            if not nucs:
                # nothing known, do not constrain
                continue
            for b in nucleosides:
                if b not in nucs:
                    # do not allow bases that are not observed in the skeleton
                    y[i][b].setInitialValue(0)
                    y[i][b].fixValue()
            if len(nucs) == 1:
                nuc = get_singleton_set_item(nucs)
                # only one base is possible, already set it to 1
                y[i][nuc].setInitialValue(1)
                y[i][nuc].fixValue()

        match self.solver:
            case "gurobi":
                solver_name = "GUROBI_CMD"
            case "cbc":
                solver_name = "PULP_CBC_CMD"

        solver = getSolver(solver_name, threads=self.threads)
        # gurobi.msg = False
        # TODO the returned value resembles the accuracy of the prediction
        _ = prob.solve(solver)

        def get_base(i):
            for b in nucleosides:
                if milp_is_one(y[i][b]):
                    return b
            return None

        def get_base_fragmentwise(i, j):
            for b in nucleosides:
                if milp_is_one(z[i][j][b]):
                    return b
            return None

        # interpret solution
        seq = [get_base(i) for i in range(self.seq_len)]
        print("Predicted sequence = ", "".join(seq))

        # Get the sequence corresponding to each of the fragments!
        fragment_seq = [
            [
                get_base_fragmentwise(i, j)
                for i in range(self.seq_len)
                if get_base_fragmentwise(i, j) is not None
            ]
            for j in valid_fragment_range
        ]

        # Get teh mass corresponding to each of the fragments!
        predicted_fragment_mass = [
            sum(
                [
                    nucleoside_masses[get_base_fragmentwise(i, j)]
                    for i in range(self.seq_len)
                    if get_base_fragmentwise(i, j) is not None
                ]
            )
            for j in valid_fragment_range
        ]

        fragment_predictions = pl.from_dicts(
            [
                {
                    # Because of the relaxation of the LP, sometimes the value is not exactly 1
                    "left": min(
                        (i for i in range(self.seq_len) if milp_is_one(x[i][j])),
                        default=0,
                    ),
                    "right": max(
                        (i for i in range(self.seq_len) if milp_is_one(x[i][j])),
                        default=-1,
                    )
                    + 1,  # right bound shall be exclusive, hence add 1
                    "predicted_fragment_seq": fragment_seq[j],
                    "predicted_fragment_mass": predicted_fragment_mass[j],
                    "observed_mass": fragment_masses[j],
                    "predicted_mass_diff": predicted_mass_diff[j].value(),
                }
                for j in valid_fragment_range
            ]
        )
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

    def _build_skeleton(
        self,
    ) -> Tuple[
        List[Set[str]],
        List[TerminalFragment],
        List[TerminalFragment],
        List[int],
        List[int],
    ]:
        skeleton_seq_start, start_fragments, invalid_start_fragments = (
            self._predict_skeleton(
                Side.START,
            )
        )

        skeleton_seq_end, end_fragments, invalid_end_fragments = self._predict_skeleton(
            Side.END,
        )
        skeleton_seq = self._align_skeletons(skeleton_seq_start, skeleton_seq_end)

        print("Skeleton sequence = ", skeleton_seq)

        return (
            skeleton_seq,
            start_fragments,
            end_fragments,
            invalid_start_fragments,
            invalid_end_fragments,
        )

    def _calculate_diffs_and_nucleosides(self):
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
        self._collect_diff_explanations()

        # TODO:
        # also consider that the observations are not complete and that we probably don't see all the letters as diffs or singletons.
        # Hence, maybe do the following: solve first with the reduced alphabet, and if the optimization does not yield a sufficiently
        # good result, then try again with an extended alphabet.
        masses = (  # self.unique_masses
            self._reduce_alphabet()
        )

        nucleosides = masses.get_column(
            "nucleoside"
        ).to_list()  # TODO: Handle the case of multiple nucleosides with the same mass when using "aggregate" grouping in the masses table
        nucleoside_masses = dict(masses.iter_rows())

        return nucleosides, nucleoside_masses

    def _collect_fragment_side_masses(
        self, side: Side, restrict_is_start_end: bool = False
    ):
        # Collects the fragment masses for the given side (also includes the start_end fragments, i.e the entire sequence)
        # Optionally ``restrict_is_start_end`` can be set to True to only consider the start_end fragments which have been included in self.fragments_side[side]
        # This is useful later when a skeleton is already built and we need the masses of the accepted fragments!

        if restrict_is_start_end:
            side_fragments = [
                i - self.mass_tags[side]
                for i in self.fragments_side[side]
                .filter(pl.col(f"is_{side}"))
                .get_column("observed_mass")
                .to_list()
            ]
            start_end_fragments = [
                i - self.mass_tags[Side.START] - self.mass_tags[Side.END]
                for i in self.fragments_side[side]
                .filter(pl.col("is_start_end"))
                .get_column("observed_mass")
                .to_list()
            ]
        else:
            # Collect the (tag subtracted) masses of the fragments for the side
            side_fragments = [
                i - self.mass_tags[side]
                for i in self.fragments.filter(pl.col(f"is_{side}"))
                .get_column("observed_mass")
                .to_list()
            ]

            # Collect the (both tags subtracted) masses of the start_end fragments
            start_end_fragments = [
                i - self.mass_tags[Side.START] - self.mass_tags[Side.END]
                for i in self.fragments.filter(pl.col("is_start_end"))
                .get_column("observed_mass")
                .to_list()
            ]

        self.fragment_masses[side] = side_fragments + start_end_fragments

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

    def _calculate_diff_dp(self, diff, threshold, explanation_masses):
        explanation_list = list(
            explain_mass(
                diff,
                explanation_masses=explanation_masses,
                matching_threshold=threshold,
            ).explanations
        )
        if len(explanation_list) > 0:
            retval = [
                Explanation(*explanation_list[i]) for i in range(len(explanation_list))
            ]
        else:
            retval = []
        return retval

    def _collect_diff_explanations(self) -> None:
        diffs = (self.mass_diffs[Side.START]) + (self.mass_diffs[Side.END])

        diffs_errors = (
            (self.mass_diffs_errors[Side.START]) + (self.mass_diffs_errors[Side.END])
        )
        for diff, diff_error in zip(diffs, diffs_errors):
            self.explanations[diff] = self._calculate_diff_dp(
                diff, diff_error, self.explanation_masses
            )

        for diff in self.singleton_masses:
            self.explanations[diff] = self._calculate_diff_dp(
                diff, self.matching_threshold, self.explanation_masses
            )

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

        return reduced

    def _predict_skeleton(
        self,
        side: Side,
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
                    diff, threshold, self.explanation_masses
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
                    # print("p_specific_explanations = ", p_specific_explanations)
                    # print("alphabet_per_expl_len = ", alphabet_per_expl_len)
                    # print("p = ", p)
                    # print("fragment_index = ", fragment_index)

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
                    # TODO can we stop ealy in building the ladder?
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
