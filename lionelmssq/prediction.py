from dataclasses import dataclass
from itertools import chain, combinations, groupby, permutations
from pathlib import Path
from typing import List, Optional, Self, Set, Tuple
from pulp import LpProblem, LpMinimize, LpInteger, LpContinuous, LpVariable, lpSum
from lionelmssq.common import Side, get_singleton_set_item
from lionelmssq.masses import UNIQUE_MASSES, EXPLANATION_MASSES, MATCHING_THRESHOLD
from lionelmssq.mass_explanation import explain_mass
import polars as pl
from loguru import logger

LP_relaxation_threshold = 0.9


@dataclass
class TerminalFragment:
    index: int
    min_end: int
    max_end: int


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
        unique_masses: pl.DataFrame = UNIQUE_MASSES,
        explanation_masses: pl.DataFrame = EXPLANATION_MASSES,
        matching_threshold: float = MATCHING_THRESHOLD,
        mass_tag_start: float = 0.0,
        mass_tag_end: float = 0.0,
    ):
        self.fragments = (
            fragments.with_row_index(name="orig_index")
            .with_columns(
                pl.when(pl.col("is_start") & pl.col("is_end"))
                .then(True)
                .otherwise(False)
                .alias("is_start_and_end")
            )
            .with_columns(
                pl.when(pl.col("is_start_and_end"))
                .then(False)
                .otherwise(pl.col("is_start"))
                .alias("is_start"),
                pl.when(pl.col("is_start_and_end"))
                .then(False)
                .otherwise(pl.col("is_end"))
                .alias("is_end"),
            )
            .sort(
                by=[
                    "single_nucleoside",
                    "is_start",
                    "is_end",
                    "is_start_and_end",
                    "observed_mass",
                ],
                descending=[True, True, True, True, False],
            )
            .with_columns(
                pl.when(pl.col("is_start_and_end"))
                .then(True)
                .otherwise(pl.col("is_start"))
                .alias("is_start"),
                pl.when(pl.col("is_start_and_end"))
                .then(True)
                .otherwise(pl.col("is_end"))
                .alias("is_end"),
            )
            .drop("is_start_and_end")
        )
        # Sort the fragments in the order of single nucleosides, start fragments, end fragments, start fragments AND end fragments, internal fragments and then by mass for each category!

        self.seq_len = seq_len
        self.solver = solver
        self.threads = threads
        self.diff_explanations = None
        self.explanations = {}
        self.mass_diffs = dict()
        self.mass_diffs_errors = dict()
        self.singleton_masses = None
        self.unique_masses = unique_masses
        self.explanation_masses = explanation_masses
        self.matching_threshold = matching_threshold
        self.mass_tags = {Side.START: mass_tag_start, Side.END: mass_tag_end}

        with pl.Config() as cfg:
            cfg.set_tbl_rows(-1)
            print(self.fragments)

    def predict(self) -> Prediction:
        # TODO: get rid of the requirement to pass the length of the sequence
        # and instead infer it from the fragments

        self._collect_singleton_masses()
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

        candidate_start_fragments = (
            self.fragments.with_row_index()
            .filter(pl.col("is_start"))
            .get_column("index")
            .to_list()
        )
        candidate_end_fragments = (
            self.fragments.with_row_index()
            .filter(pl.col("is_end"))
            .get_column("index")
            .to_list()
        )

        skeleton_seq, start_fragments = self._predict_skeleton(Side.START)
        _, end_fragments = self._predict_skeleton(Side.END, skeleton_seq=skeleton_seq)

        print("Skeleton sequence = ", skeleton_seq)

        prob = LpProblem("RNA sequencing", LpMinimize)
        # i = 1,...,n: positions in the sequence
        # j = 1,...,m: fragments
        # b = 1,...,k: (modified) bases

        fragment_masses = self.fragments.get_column("observed_mass").to_list()
        n_fragments = len(fragment_masses)
        nucleosides = masses.get_column(
            "nucleoside"
        ).to_list()  # TODO: Handle the case of multiple nucleosides with the same mass when using "aggregate" grouping in the masses table
        nucleoside_masses = dict(masses.iter_rows())

        if not start_fragments:
            logger.warning(
                "No start fragments provided, this will likely lead to suboptimal results."
            )

        if not end_fragments:
            logger.warning(
                "No end fragments provided, this will likely lead to suboptimal results."
            )

        # x: binary variables indicating fragment j presence at position i
        x = [
            [
                LpVariable(f"x_{i},{j}", lowBound=0, upBound=1, cat=LpInteger)
                for j in range(n_fragments)
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
                for j in range(n_fragments)
            ]
            for i in range(self.seq_len)
        ]
        # weight_diff_abs: absolute value of weight_diff
        predicted_mass_diff_abs = [
            LpVariable(f"predicted_mass_diff_abs_{j}", lowBound=0, cat=LpContinuous)
            for j in range(n_fragments)
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
            for j in range(n_fragments)
        ]
        # TODO: Why can't the abs sum be used directly in the optimization function? Why such a complication!

        # optimization function
        prob += lpSum([predicted_mass_diff_abs[j] for j in range(n_fragments)])

        # select one base per position
        for i in range(self.seq_len):
            prob += lpSum([y[i][b] for b in nucleosides]) == 1

        # fill z with the product of binary variables x and y
        for i in range(self.seq_len):
            for j in range(n_fragments):
                for b in nucleosides:
                    prob += z[i][j][b] <= x[i][j]
                    prob += z[i][j][b] <= y[i][b]
                    prob += z[i][j][b] >= x[i][j] + y[i][b] - 1

        # ensure that fragment is aligned continuously
        # (no gaps: if x[i1,j] = 1 and x[i2,j] = 1, then x[i_between,j] = 1)
        for j in range(n_fragments):
            for i1, i2 in combinations(range(self.seq_len), 2):
                # i2 and i1 are inclusive
                assert i2 > i1
                if i2 - i1 > 1:
                    prob += (x[i1][j] + x[i2][j] - 1) * (i2 - i1 - 1) <= lpSum(
                        [x[i_between][j] for i_between in range(i1 + 1, i2)]
                    )

        # ensure that start fragments are aligned at the beginning of the sequence
        for fragment in start_fragments:
            j = candidate_start_fragments[fragment.index]
            # print("End fragment = ", fragment, "Index = ", j)
            # min_end is exclusive
            for i in range(fragment.min_end):
                x[i][j].setInitialValue(1)
                x[i][j].fixValue()
            for i in range(fragment.max_end, self.seq_len):
                x[i][j].setInitialValue(0)
                x[i][j].fixValue()

        # ensure that end fragments are aligned at the end of the sequence
        for fragment in end_fragments:
            j = candidate_end_fragments[fragment.index]
            if (
                j not in [candidate_start_fragments[f.index] for f in start_fragments]
            ):  # Exlude fragments that are both start and end fragments, they were already considered with start fragments!
                # print("End fragment = ", fragment, "Index = ", j)
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
        for j in range(n_fragments):
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

        import pulp

        match self.solver:
            case "gurobi":
                solver_name = "GUROBI_CMD"
            case "cbc":
                solver_name = "PULP_CBC_CMD"

        solver = pulp.getSolver(solver_name, threads=self.threads)
        # gurobi.msg = False
        # TODO the returned value resembles the accuracy of the prediction
        _ = prob.solve(solver)

        def get_base(i):
            for b in nucleosides:
                # if y[i][b].value() == 1:
                if y[i][b].value() > LP_relaxation_threshold:
                    return b
            return None

        def get_base_fragmentwise(i, j):
            for b in nucleosides:
                if z[i][j][b].value() > LP_relaxation_threshold:
                    return b
            return None

        # interpret solution
        seq = [get_base(i) for i in range(self.seq_len)]
        print("Predicted sequence = ", "".join(seq))
        fragment_seq = [
            [
                get_base_fragmentwise(i, j)
                for i in range(self.seq_len)
                if get_base_fragmentwise(i, j) is not None
            ]
            for j in range(n_fragments)
        ]
        # print("Predicted fragment sequence = ", fragment_seq)
        predicted_fragment_mass = [
            sum(
                [
                    nucleoside_masses[get_base_fragmentwise(i, j)]
                    for i in range(self.seq_len)
                    if get_base_fragmentwise(i, j) is not None
                ]
            )
            for j in range(n_fragments)
        ]

        fragment_predictions = pl.from_dicts(
            [
                {
                    # Because of the relaxation of the LP, sometimes the value is not exactly 1
                    "left": min(
                        (
                            i
                            for i in range(self.seq_len)
                            if x[i][j].value() > LP_relaxation_threshold
                        ),
                        default=0,
                    ),
                    "right": max(
                        (
                            i
                            for i in range(self.seq_len)
                            if x[i][j].value() > LP_relaxation_threshold
                        ),
                        default=-1,
                    )
                    + 1,  # right bound shall be exclusive, hence add 1
                    "predicted_fragment_seq": fragment_seq[j],
                    "predicted_fragment_mass": predicted_fragment_mass[j],
                    "observed_mass": fragment_masses[j],
                    "predicted_mass_diff": predicted_mass_diff[j].value(),
                }
                for j in range(n_fragments)
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

    def _collect_diffs(self, side: Side) -> None:
        masses = self.fragments.filter(pl.col(f"is_{side}")).get_column("observed_mass")
        self.mass_diffs[side] = [masses[0]] + (masses[1:] - masses[:-1]).to_list()
        self.mass_diffs_errors[side] = [
            self._calculate_diff_errors(
                self.mass_tags[side],
                masses[0] + self.mass_tags[side],
                self.matching_threshold,
            )
        ] + (
            [
                self._calculate_diff_errors(
                    masses[i] + self.mass_tags[side], masses[i - 1] + self.mass_tags[side], self.matching_threshold
                )
                for i in range(1, len(masses))
            ]
        )

    def _calculate_diff_errors(self, mass1, mass2, threshold) -> float:

        retval =  threshold * ((mass1**2 + mass2**2) ** 0.5) / abs(mass1 - mass2)
        # Constrain the maximum relative error to 1! 
        #For mass difference very close to zero, the relative error can be very high!
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
        temp = list(
            explain_mass(
                diff,
                explanation_masses=explanation_masses,
                matching_threshold=threshold,
            ).explanations
        )
        if len(temp) > 0:
            retval = [Explanation(*temp[i]) for i in range(len(temp))]
        else:
            retval = []
        return retval

    def _collect_diff_explanations(self) -> None:
        diffs = (self.mass_diffs[Side.START]) + (self.mass_diffs[Side.END])

        diffs_errors = (
            (self.mass_diffs_errors[Side.START]) + (self.mass_diffs_errors[Side.END])
        )
        for diff, diff_error in zip(diffs, diffs_errors):
            self.explanations[diff] = self._calculate_diff_dp(diff, diff_error, self.explanation_masses)

        for diff in self.singleton_masses:
            self.explanations[diff] = self._calculate_diff_dp(
                diff, self.matching_threshold, self.explanation_masses
            )
        #TODO: Can make it simpler here by rejecting diff which cannot be explained instead of doing it in the _predict_skeleton function!

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
        self, side: Side, skeleton_seq: Optional[List[Set[str]]] = None
    ) -> Tuple[List[Set[str]], List[TerminalFragment]]:
        if skeleton_seq is None:
            skeleton_seq = [set() for _ in range(self.seq_len)]

        factor = 1 if side == Side.START else -1

        masses = self.fragments.filter(pl.col(f"is_{side}")).get_column("observed_mass")

        def get_possible_nucleosides(pos: int, i: int) -> Set[str]:
            return skeleton_seq[pos + factor * i]

        def is_valid_pos(pos: int, ext: int) -> bool:
            pos = pos + factor * ext
            return (
                0 <= pos <= self.seq_len
                if side == Side.START
                else -(self.seq_len + 1) <= pos < 0
                # else -self.seq_len <= pos < 0
            )

        pos = {0} if side == Side.START else {-1}
        carry_over_mass = 0

        valid_terminal_fragments = []
        last_mass_valid = 0.
        for fragment_index, (diff, mass) in enumerate(zip(self.mass_diffs[side], masses)):
            diff += carry_over_mass
            assert pos

            is_valid = True
            if diff in self.explanations:
                explanations = self.explanations.get(diff, [])
                # last_mass_valid = mass
            else:
                #TODO: Review if the correct values of the mass, i.e "last_mass_valid" are used here!
                threshold = self._calculate_diff_errors(last_mass_valid+self.mass_tags[side],last_mass_valid+diff+self.mass_tags[side],self.matching_threshold)
                explanations = self._calculate_diff_dp(
                diff, threshold, self.explanation_masses
                )
                if explanations:
                    carry_over_mass = 0

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
                        if side is Side.START:
                            if p == min_p:
                                min_fragment_end = min_p + min(
                                    expl_len for expl_len in alphabet_per_expl_len
                                )
                            elif p == max_p:
                                max_fragment_end = max_p + max(
                                    expl_len for expl_len in alphabet_per_expl_len
                                )
                        else:
                            if p == min_p:
                                min_fragment_end = min_p - min(
                                    expl_len for expl_len in alphabet_per_expl_len
                                )
                            elif p == max_p:
                                max_fragment_end = max_p - max(
                                    expl_len for expl_len in alphabet_per_expl_len
                                )

                    # constrain already existing sets in the range of the expl
                    # by the nucs that are given in the explanations
                    for expl_len, alphabet in alphabet_per_expl_len.items():
                        for i in range(expl_len):
                            possible_nucleosides = get_possible_nucleosides(p, i)
                            if possible_nucleosides.issuperset(alphabet):
                                # the expl sharpens the possibilities
                                # clear the possibilities so far, the expl will add
                                # the sharpened ones ones below
                                possible_nucleosides.clear()

                    for expl in p_specific_explanations:
                        for perm in permutations(expl):
                            for i, nuc in enumerate(perm):
                                get_possible_nucleosides(p, i).add(nuc)

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
                abs(diff) <= self.matching_threshold*abs(mass+self.mass_tags[side]) 
                #abs(diff/mass) <= self._calculate_diff_errors(mass+self.mass_tags[side],mass+diff+self.mass_tags[side],self.matching_threshold) 
                #Problem! The above approach might blow up if the masses are very close, i.e. diff is very close to zero!
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
                        index=fragment_index,
                        min_end=min_fragment_end,
                        max_end=max_fragment_end,
                    )
                )
                pos = next_pos
                last_mass_valid = mass #TODO:Review this!
            else:  # TODO: IMP: Consider the skipped fragments as internal fragments! Need to add back the terminal mass to this fragments! This being done, but the terminal mass is not added back to these fragments!
                logger.warning(
                    f"Skipping {side} fragment {fragment_index} because no "
                    "explanations are found for the mass difference."
                )
                carry_over_mass += diff
        return skeleton_seq, valid_terminal_fragments

    # TODO: While building the ladder it may happen that things are unambiguous from one side, but not from the other!
    # In that case, we should consider the unambiguous side as the correct one and the ambiguous side as the one to be fixed!


class Explanation:
    def __init__(self, *nucleosides):
        self.nucleosides = tuple(sorted(nucleosides))

    def __iter__(self):
        yield from self.nucleosides

    def __len__(self):
        return len(self.nucleosides)

    def __repr__(self):
        return f"{{{",".join(self.nucleosides)}}}"
