from pulp import (
    LpProblem,
    LpMinimize,
    LpInteger,
    LpContinuous,
    LpVariable,
    lpSum,
    getSolver,
)
from lionelmssq.common import get_singleton_set_item, milp_is_one
from itertools import chain, combinations, groupby
import polars as pl


class LinearProgramInstance:
    def __init__(self, fragment_masses, start_fragments, end_fragments,
                 seq_len, fragments, nucleosides, nucleoside_masses,
                 skeleton_seq):
        # i = 1,...,n: positions in the sequence
        # j = 1,...,m: fragments
        # b = 1,...,k: (modified) bases
        self.fragment_masses = fragment_masses
        self.seq_len = seq_len
        self.nucleosides = nucleosides
        self.nucleoside_masses = nucleoside_masses
        valid_fragment_range = list(range(len(fragment_masses)))
        # x: binary variables indicating fragment j presence at position i
        self.x = self._set_x(valid_fragment_range, start_fragments,
                             end_fragments, fragments)
        # y: binary variables indicating base b at position i
        self.y = self._set_y(skeleton_seq)
        # z: binary variables indicating product of x and y
        self.z = self._set_z(valid_fragment_range)
        # weight_diff: difference between fragment monoisotopic mass and sum of masses of bases in fragment as estimated in the MILP
        self.predicted_mass_diff = self._set_predicted_mass_difference(
            valid_fragment_range)

        self.problem = self._define_lp_problem(valid_fragment_range)

    def _set_x(self, valid_fragment_range, start_fragments,
               end_fragments, fragments):
        x = [
            [
                LpVariable(f"x_{i},{j}", lowBound=0, upBound=1,
                           cat=LpInteger)
                for j in valid_fragment_range
            ]
            for i in range(self.seq_len)
        ]

        # ensure that start fragments are aligned at the beginning of the sequence
        for fragment in start_fragments:
            # j is the row index where the "index" matches fragment.index
            # fragment.index uses the original (mass sorted) index of the read fragment files,
            # but in self.fragments we disqualify many fragments of the original file.
            # Hence, we need to find the correct row index in self.fragments which corresponds to the original index
            # since in the MILP we fit all the fragments of self.fragments
            j = (
                fragments.with_row_index("row_index")
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
                fragments.with_row_index("row_index")
                .filter(pl.col("index") == fragment.index)
                .item(0, "row_index")
            )
            # min_end is exclusive
            for i in range(fragment.min_end+1, 0):
                x[i][j].setInitialValue(1)
                x[i][j].fixValue()
            for i in range(-self.seq_len, fragment.max_end+1):
                x[i][j].setInitialValue(0)
                x[i][j].fixValue()

        # Fragments that aren't either start or end are either inner or uncertain.
        # Hence, we don't further constrain their positioning and length and let the
        # LP decide.

        return x

    def _set_y(self, skeleton_seq):
        y = [
            {
                b: LpVariable(f"y_{i},{b}", lowBound=0, upBound=1,
                              cat=LpInteger)
                for b in self.nucleosides
            }
            for i in range(self.seq_len)
        ]

        # use skeleton seq to fix bases
        for i, nucs in enumerate(skeleton_seq):
            if not nucs:
                # nothing known, do not constrain
                continue
            for b in self.nucleosides:
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
                        f"z_{i},{j},{b}", lowBound=0, upBound=1,
                        cat=LpInteger
                    )
                    for b in self.nucleosides
                }
                for j in valid_fragment_range
            ]
            for i in range(self.seq_len)
        ]
        return z

    def _set_predicted_mass_difference(self, valid_fragment_range):
        return [
            self.fragment_masses[j]
            -lpSum(
                [
                    self.z[i][j][b] * self.nucleoside_masses[b]
                    for i in range(self.seq_len)
                    for b in self.nucleosides
                ]
            )
            for j in valid_fragment_range
        ]

    def _define_lp_problem(self, valid_fragment_range):
        problem = LpProblem("Fragment filter", LpMinimize)

        # weight_diff_abs: absolute value of weight_diff
        predicted_mass_diff_abs = [
            LpVariable(f"predicted_mass_diff_abs_{j}", lowBound=0,
                       cat=LpContinuous)
            for j in valid_fragment_range
        ]

        # optimization function
        problem += lpSum(
            [predicted_mass_diff_abs[j] for j in valid_fragment_range])

        # select one base per position
        for i in range(self.seq_len):
            problem += lpSum([self.y[i][b] for b in self.nucleosides]) == 1

        # fill z with the product of binary variables x and y
        for i in range(self.seq_len):
            for j in valid_fragment_range:
                for b in self.nucleosides:
                    problem += self.z[i][j][b] <= self.x[i][j]
                    problem += self.z[i][j][b] <= self.y[i][b]
                    problem += self.z[i][j][b] >= self.x[i][j]+self.y[i][b]-1

        # ensure that fragment is aligned continuously
        # (no gaps: if x[i1,j] = 1 and x[i2,j] = 1, then x[i_between,j] = 1)
        for j in valid_fragment_range:
            for i1, i2 in combinations(range(self.seq_len), 2):
                # i2 and i1 are inclusive
                assert i2 > i1
                if i2-i1 > 1:
                    problem += ((self.x[i1][j]+self.x[i2][j]-1) * (i2-i1-1) <=
                                lpSum(
                        [self.x[i_between][j] for i_between in range(i1+1, i2)]
                    ))


        # constrain weight_diff_abs to be the absolute value of weight_diff
        for j in valid_fragment_range:
            # if j not in invalid_start_fragments and j not in invalid_end_fragments:
            problem += (predicted_mass_diff_abs[j] >=
                        self.predicted_mass_diff[j])
            problem += (predicted_mass_diff_abs[j] >=
                        -self.predicted_mass_diff[j])

        return problem

    def check_feasibility(self, solver_name, num_threads, threshold):
        solver = getSolver(solver_name, threads=num_threads, msg=False)
        _ = self.problem.solve(solver)
        return self.problem.objective.value() <= threshold

    def evaluate(self, solver_name, num_threads):
        solver = getSolver(solver_name, threads=num_threads, msg=False)
        # gurobi.msg = False
        # TODO the returned value resembles the accuracy of the prediction
        _ = self.problem.solve(solver)

        # interpret solution
        seq = [self._get_base(i) for i in range(self.seq_len)]
        print("Predicted sequence = ", "".join(seq))

        # Get the sequence corresponding to each of the fragments!
        fragment_seq = [
            [
                self._get_base_fragmentwise(i, j)
                for i in range(self.seq_len)
                if self._get_base_fragmentwise(i, j) is not None
            ]
            for j in list(range(len(self.fragment_masses)))
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
            for j in list(range(len(self.fragment_masses)))
        ]

        fragment_predictions = pl.from_dicts(
            [
                {
                    # Because of the relaxation of the LP, sometimes the value is not exactly 1
                    "left": min(
                        (i for i in range(self.seq_len) if
                         milp_is_one(self.x[i][j])),
                        default=0,
                    ),
                    "right": max(
                        (i for i in range(self.seq_len) if
                         milp_is_one(self.x[i][j])),
                        default=-1,
                    )
                             +1,  # right bound shall be exclusive, hence add 1
                    "predicted_fragment_seq": fragment_seq[j],
                    "predicted_fragment_mass": predicted_fragment_mass[j],
                    "observed_mass": self.fragment_masses[j],
                    "predicted_mass_diff": self.predicted_mass_diff[j].value(),
                }
                for j in list(range(len(self.fragment_masses)))
            ]
        )

        return seq, fragment_predictions

    def _get_base(self, i):
        for b in self.nucleosides:
            if milp_is_one(self.y[i][b]):
                return b
        return None

    def _get_base_fragmentwise(self, i, j):
        for b in self.nucleosides:
            if milp_is_one(self.z[i][j][b]):
                return b
        return None
