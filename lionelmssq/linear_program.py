from pulp import (
    LpProblem,
    LpMinimize,
    LpInteger,
    LpContinuous,
    LpVariable,
    lpSum,
    getSolver,
)
from typing import Any, Set
from itertools import combinations
import polars as pl
import numpy as np

from lionelmssq.masses import UNMODIFIED_BASES


MILP_QUASI_ONE_THRESHOLD = 0.9


def milp_is_one(var, threshold=MILP_QUASI_ONE_THRESHOLD):
    # Due to the LP relaxation, the LP sometimes does not exactly output
    # probabilities of 1 for one nucleotide or one position.
    # Hence, we need to set a threshold for the LP relaxation.
    return var.value() >= threshold


def get_singleton_set_item(set_: Set[Any]) -> Any:
    """Return the only item in a set."""
    if len(set_) != 1:
        raise ValueError(f"Expected a set with one item, got {set_}")
    return next(iter(set_))


class LinearProgramInstance:
    def __init__(
        self, fragments, nucleosides, dp_table, skeleton_seq, modification_rate
    ):
        # i = 1,...,n: positions in the sequence
        # j = 1,...,m: fragments
        # b = 1,...,k: (modified) bases
        self.fragments = fragments
        self.seq_len = len(skeleton_seq)
        self.nucleoside_names = nucleosides.get_column("nucleoside").to_list()
        self.nucleoside_masses = dict(
            nucleosides.select(["nucleoside", "standard_unit_mass"]).iter_rows()
        )

        fragment_masses = self.fragments.get_column("standard_unit_mass").to_list()
        valid_fragment_range = list(range(len(fragment_masses)))

        # x: binary variables indicating fragment j presence at position i
        self.x = self._set_x(valid_fragment_range, fragments)
        # y: binary variables indicating base b at position i
        self.y = self._set_y(skeleton_seq)
        # z: binary variables indicating product of x and y
        self.z = self._set_z(valid_fragment_range)

        # weight_diff: difference between fragment monoisotopic mass and sum of masses of bases in fragment as estimated in the MILP
        self.predicted_mass_diff = self._set_predicted_mass_difference(
            fragment_masses, valid_fragment_range
        )

        self.problem = self._define_lp_problem(
            valid_fragment_range, modification_rate, dp_table
        )

    def _set_x(self, valid_fragment_range, fragments):
        x = [
            [
                LpVariable(f"x_{i},{j}", lowBound=0, upBound=1, cat=LpInteger)
                for j in valid_fragment_range
            ]
            for i in range(self.seq_len)
        ]

        for j in range(len(fragments)):
            # Ensure complete fragments are aligned at the whole sequence
            if fragments.item(j, "breakage") == "START_END":
                for i in range(self.seq_len):
                    x[i][j].setInitialValue(1)
                    x[i][j].fixValue()
                continue

            # Ensure START fragments are aligned at the beginning of the sequence
            if "START" in fragments.item(j, "breakage"):
                # min_end is exclusive
                for i in range(fragments.item(j, "min_end")):
                    x[i][j].setInitialValue(1)
                    x[i][j].fixValue()
                for i in range(fragments.item(j, "max_end"), self.seq_len):
                    x[i][j].setInitialValue(0)
                    x[i][j].fixValue()
                continue

            # Ensure END fragments are aligned at the end of the sequence
            if "END" in fragments.item(j, "breakage"):
                # min_end is exclusive
                for i in range(fragments.item(j, "min_end") + 1, 0):
                    x[i][j].setInitialValue(1)
                    x[i][j].fixValue()
                for i in range(-self.seq_len, fragments.item(j, "max_end") + 1):
                    x[i][j].setInitialValue(0)
                    x[i][j].fixValue()
                continue

            # Internal fragments are not further constrained in both their
            # positioning and length for now; let the LP decide.

        return x

    def _set_y(self, skeleton_seq):
        y = [
            {
                b: LpVariable(f"y_{i},{b}", lowBound=0, upBound=1, cat=LpInteger)
                for b in self.nucleoside_names
            }
            for i in range(self.seq_len)
        ]

        # use skeleton seq to fix bases
        for i, nucs in enumerate(skeleton_seq):
            if not nucs:
                # nothing known, do not constrain
                continue
            for b in self.nucleoside_names:
                if b not in nucs:
                    # do not allow bases that are not observed in the skeleton
                    y[i][b].setInitialValue(0)
                    y[i][b].fixValue()
            if len(nucs) == 1:
                nuc = get_singleton_set_item(nucs)
                # only one base is possible, already set it to 1
                y[i][nuc].setInitialValue(1)
                y[i][nuc].fixValue()

        return y

    def _set_z(self, valid_fragment_range):
        z = [
            [
                {
                    b: LpVariable(
                        f"z_{i},{j},{b}", lowBound=0, upBound=1, cat=LpInteger
                    )
                    for b in self.nucleoside_names
                }
                for j in valid_fragment_range
            ]
            for i in range(self.seq_len)
        ]
        return z

    def _set_predicted_mass_difference(self, fragment_masses, valid_fragment_range):
        return [
            fragment_masses[j]
            - lpSum(
                [
                    self.z[i][j][b] * self.nucleoside_masses[b]
                    for i in range(self.seq_len)
                    for b in self.nucleoside_names
                ]
            )
            for j in valid_fragment_range
        ]

    def _define_lp_problem(self, valid_fragment_range, modification_rate, dp_table):
        problem = LpProblem("fragment_filter", LpMinimize)

        # weight_diff_abs: absolute value of weight_diff
        predicted_mass_diff_abs = [
            LpVariable(f"predicted_mass_diff_abs_{j}", lowBound=0, cat=LpContinuous)
            for j in valid_fragment_range
        ]

        # optimization function
        problem += lpSum([predicted_mass_diff_abs[j] for j in valid_fragment_range])

        # select one base per position
        for i in range(self.seq_len):
            problem += lpSum([self.y[i][b] for b in self.nucleoside_names]) == 1

        # Enforce universal modification rate
        problem += lpSum(
            [
                self.y[i][b]
                for i in range(self.seq_len)
                for b in self.nucleoside_names
                if b not in UNMODIFIED_BASES
            ]
        ) <= np.ceil(modification_rate * self.seq_len)

        # Enforce individual modification rates
        for mass in dp_table.masses:
            for b in mass.names:
                if b in self.nucleoside_names:
                    problem += lpSum(
                        [self.y[i][b] for i in range(self.seq_len)]
                    ) <= np.ceil(mass.modification_rate * self.seq_len)

        # fill z with the product of binary variables x and y
        for i in range(self.seq_len):
            for j in valid_fragment_range:
                for b in self.nucleoside_names:
                    problem += self.z[i][j][b] <= self.x[i][j]
                    problem += self.z[i][j][b] <= self.y[i][b]
                    problem += self.z[i][j][b] >= self.x[i][j] + self.y[i][b] - 1

        # ensure that fragment is aligned continuously
        # (no gaps: if x[i1,j] = 1 and x[i2,j] = 1, then x[i_between,j] = 1)
        for j in valid_fragment_range:
            for i1, i2 in combinations(range(self.seq_len), 2):
                # i2 and i1 are inclusive
                assert i2 > i1
                if i2 - i1 > 1:
                    problem += (self.x[i1][j] + self.x[i2][j] - 1) * (
                        i2 - i1 - 1
                    ) <= lpSum(
                        [self.x[i_between][j] for i_between in range(i1 + 1, i2)]
                    )

        # constrain weight_diff_abs to be the absolute value of weight_diff
        for j in valid_fragment_range:
            # if j not in invalid_start_fragments and j not in invalid_end_fragments:
            problem += predicted_mass_diff_abs[j] >= self.predicted_mass_diff[j]
            problem += predicted_mass_diff_abs[j] >= -self.predicted_mass_diff[j]

        return problem

    def check_feasibility(self, solver_params, threshold):
        solver = getSolver(**solver_params, timeLimit=30)
        _ = self.problem.solve(solver)
        return self.problem.objective.value() <= threshold

    def evaluate(self, solver_params):
        solver = getSolver(**solver_params, timeLimit=300)
        # gurobi.msg = False
        # TODO the returned value resembles the accuracy of the prediction
        _ = self.problem.solve(solver)

        # interpret solution
        seq = [self._get_base(i) for i in range(self.seq_len)]
        print(
            "Predicted sequence = ",
            "".join([val if val is not None else "-" for val in seq]),
        )

        fragment_masses = self.fragments.get_column("standard_unit_mass").to_list()

        # Get the sequence corresponding to each of the fragments!
        fragment_seq = [
            "".join(
                [
                    self._get_base_fragmentwise(i, j)
                    for i in range(self.seq_len)
                    if self._get_base_fragmentwise(i, j) is not None
                ]
            )
            for j in list(range(len(fragment_masses)))
        ]

        # Get the mass corresponding to each of the fragments!
        predicted_fragment_mass = [
            sum(
                [
                    self.nucleoside_masses[self._get_base_fragmentwise(i, j)]
                    for i in range(self.seq_len)
                    if self._get_base_fragmentwise(i, j) is not None
                ]
            )
            for j in list(range(len(fragment_masses)))
        ]

        observed_masses = self.fragments.get_column("observed_mass").to_list()
        fragment_predictions = pl.from_dicts(
            [
                {
                    # Because of the relaxation of the LP, sometimes the value is not exactly 1
                    "left": min(
                        (i for i in range(self.seq_len) if milp_is_one(self.x[i][j])),
                        default=0,
                    ),
                    "right": max(
                        (i for i in range(self.seq_len) if milp_is_one(self.x[i][j])),
                        default=-1,
                    )
                    + 1,  # right bound shall be exclusive, hence add 1
                    "observed_mass": observed_masses[j],
                    "standard_unit_mass": fragment_masses[j],
                    "predicted_mass": predicted_fragment_mass[j],
                    "predicted_diff": self.predicted_mass_diff[j].value(),
                    "predicted_seq": fragment_seq[j],
                }
                for j in list(range(len(fragment_masses)))
            ]
        )

        fragment_predictions = pl.concat(
            [fragment_predictions, self.fragments.select(pl.col("orig_index"))],
            how="horizontal",
        )

        # reorder fragment predictions so that they match the original order again
        fragment_predictions = fragment_predictions.sort("orig_index")

        return seq, fragment_predictions

    def _get_base(self, i):
        for b in self.nucleoside_names:
            if milp_is_one(self.y[i][b]):
                return b
        return None

    def _get_base_fragmentwise(self, i, j):
        for b in self.nucleoside_names:
            if milp_is_one(self.z[i][j][b]):
                return b
        return None
