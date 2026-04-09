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

from spectrseqtools.dataclasses import SolverParameters
from spectrseqtools.masses import UNMODIFIED_BASES
from spectrseqtools.prediction.traceback_matrix import CompositionInferrer


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
    def __init__(self, fragments, inferrer: CompositionInferrer, skeleton_seq):
        # i = 1,...,N: (modified) nucleotides
        # j = 1,...,M: fragments
        # k = 1,...,S: positions in the sequence
        self.fragments = fragments
        self.seq_len = len(skeleton_seq)
        self.nucleoside_names = [mass.names[0] for mass in inferrer.alphabet[1:]]
        self.nucleoside_masses = {
            mass.names[0]: mass.mass * inferrer.precision
            for mass in inferrer.alphabet[1:]
        }

        fragment_masses = self.fragments.get_column("standard_unit_mass").to_list()
        valid_fragment_range = list(range(len(fragment_masses)))

        # x: binary variables indicating fragment j presence at position k
        self.x = self._set_x(valid_fragment_range, fragments)
        # y: binary variables indicating nucleotide i at position k
        self.y = self._set_y(skeleton_seq)
        # z: binary variables indicating product of x and y
        self.z = self._set_z(valid_fragment_range)

        # predicted_mass_diff: difference between a fragment's SU-mass
        # and the sum of nucleotide masses assigned by the MILP prediction
        self.predicted_mass_diff = self._set_predicted_mass_difference(
            fragment_masses, valid_fragment_range
        )

        self.problem = self._define_lp_problem(valid_fragment_range, inferrer)

    def _set_x(self, valid_fragment_range, fragments):
        x = [
            [
                LpVariable(f"x_{j},{k}", lowBound=0, upBound=1, cat=LpInteger)
                for k in range(self.seq_len)
            ]
            for j in valid_fragment_range
        ]

        for j in range(len(fragments)):
            # Ensure intact fragments are aligned at the whole sequence
            if fragments.item(j, "fragmentation") == "START_END":
                for k in range(self.seq_len):
                    x[j][k].setInitialValue(1)
                    x[j][k].fixValue()
                continue

            # Ensure START fragments are aligned at the beginning of the sequence
            if "START" in fragments.item(j, "fragmentation"):
                # min_end is exclusive
                for k in range(fragments.item(j, "min_end") + 1):
                    x[j][k].setInitialValue(1)
                    x[j][k].fixValue()
                for k in range(fragments.item(j, "max_end") + 1, self.seq_len):
                    x[j][k].setInitialValue(0)
                    x[j][k].fixValue()
                continue

            # Ensure END fragments are aligned at the end of the sequence
            if "END" in fragments.item(j, "fragmentation"):
                # min_end is exclusive
                for k in range(fragments.item(j, "max_end")):
                    x[j][k].setInitialValue(0)
                    x[j][k].fixValue()
                for k in range(fragments.item(j, "min_end"), self.seq_len):
                    x[j][k].setInitialValue(1)
                    x[j][k].fixValue()
                continue

            # Internal fragments are not further constrained in both their
            # positioning and length for now; let the LP decide.

        return x

    def _set_y(self, skeleton_seq):
        y = [
            [
                LpVariable(f"y_{i},{k}", lowBound=0, upBound=1, cat=LpInteger)
                for k in range(self.seq_len)
            ]
            for i in range(len(self.nucleoside_names))
        ]

        # Use skeleton sequence to fix nucleotides
        for k, nucs in enumerate(skeleton_seq):
            if not nucs:
                # Do not constrain if nothing is known
                continue
            for i, nuc in enumerate(self.nucleoside_names):
                # Do not allow nucleotides that are not observed in the skeleton
                if nuc not in nucs:
                    y[i][k].setInitialValue(0)
                    y[i][k].fixValue()
                # If only one nucleotide is possible, fix the value already
                if len(nucs) == 1:
                    if nuc == get_singleton_set_item(nucs):
                        y[i][k].setInitialValue(1)
                        y[i][k].fixValue()

        return y

    def _set_z(self, valid_fragment_range):
        z = [
            [
                [
                    LpVariable(f"z_{i},{j},{k}", lowBound=0, upBound=1, cat=LpInteger)
                    for k in range(self.seq_len)
                ]
                for j in valid_fragment_range
            ]
            for i in range(len(self.nucleoside_names))
        ]
        return z

    def _set_predicted_mass_difference(self, fragment_masses, valid_fragment_range):
        return [
            fragment_masses[j]
            - lpSum(
                [
                    self.z[i][j][k] * self.nucleoside_masses[nuc]
                    for i, nuc in enumerate(self.nucleoside_names)
                    for k in range(self.seq_len)
                ]
            )
            for j in valid_fragment_range
        ]

    def _define_lp_problem(self, valid_fragment_range, inferrer):
        problem = LpProblem("fragment_filter", LpMinimize)

        # predicted_mass_diff_abs: absolute value of predicted_mass_diff
        predicted_mass_diff_abs = [
            LpVariable(f"predicted_mass_diff_abs_{j}", lowBound=0, cat=LpContinuous)
            for j in valid_fragment_range
        ]

        # Set optimization function
        problem += lpSum([predicted_mass_diff_abs[j] for j in valid_fragment_range])

        # Select one nucleotide per position
        for k in range(self.seq_len):
            problem += (
                lpSum([self.y[i][k] for i in range(len(self.nucleoside_names))]) == 1
            )

        # Enforce universal modification rate
        problem += lpSum(
            [
                self.y[i][k]
                for k in range(self.seq_len)
                for i, nuc in enumerate(self.nucleoside_names)
                if nuc not in UNMODIFIED_BASES
            ]
        ) <= np.ceil(inferrer.seq.modification_rate * self.seq_len)

        # Enforce individual modification rates
        for mass in inferrer.alphabet:
            for i in mass.names:
                if i in range(len(self.nucleoside_names)):
                    problem += lpSum(
                        [self.y[i][k] for k in range(self.seq_len)]
                    ) <= np.ceil(mass.modification_rate * self.seq_len)

        # Fill z with the product of binary variables x and y
        for k in range(self.seq_len):
            for j in valid_fragment_range:
                for i in range(len(self.nucleoside_names)):
                    problem += self.z[i][j][k] <= self.x[j][k]
                    problem += self.z[i][j][k] <= self.y[i][k]
                    problem += self.z[i][j][k] >= self.x[j][k] + self.y[i][k] - 1

        # Ensure that fragments are aligned continuously (i.e. no gaps:
        # if x[j, k1] = 1 and x[j, k2] = 1, then x[j, k_between] = 1)
        for j in valid_fragment_range:
            for k1, k2 in combinations(range(self.seq_len), 2):
                # k1 and k2 are inclusive
                assert k2 > k1
                if k2 - k1 > 1:
                    problem += (self.x[j][k1] + self.x[j][k2] - 1) * (
                        k2 - k1 - 1
                    ) <= lpSum(
                        [self.x[j][k_between] for k_between in range(k1 + 1, k2)]
                    )

        # Constrain predicted_mass_diff_abs to be the absolute value of predicted_mass_diff
        for j in valid_fragment_range:
            problem += predicted_mass_diff_abs[j] >= self.predicted_mass_diff[j]
            problem += predicted_mass_diff_abs[j] >= -self.predicted_mass_diff[j]

        return problem

    def minimize_error(self, solver_params: SolverParameters) -> float:
        # Initialize solver
        solver = getSolver(**solver_params.to_dict(filter_only=True))

        _ = self.problem.solve(solver)
        score = self.problem.objective.value()
        return np.inf if score is None else score

    def evaluate(self, solver_params: SolverParameters):
        # Initialize solver
        solver = getSolver(**solver_params.to_dict(filter_only=False))

        # TODO: Make returned value resemble prediction accuracy
        _ = self.problem.solve(solver)

        # Interpret solution
        seq = [self._get_sequence_nucleotide(k) for k in range(self.seq_len)]

        fragment_masses = self.fragments.get_column("standard_unit_mass").to_list()

        # Get the sequence corresponding to each of the fragments!
        fragment_seq = [
            "".join(
                [
                    self._get_fragment_nucleotide(j, k)
                    for k in range(self.seq_len)
                    if self._get_fragment_nucleotide(j, k) is not None
                ]
            )
            for j in list(range(len(fragment_masses)))
        ]

        # Get the mass corresponding to each of the fragments!
        predicted_fragment_mass = [
            sum(
                [
                    self.nucleoside_masses[self._get_fragment_nucleotide(j, k)]
                    for k in range(self.seq_len)
                    if self._get_fragment_nucleotide(j, k) is not None
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
                        (k for k in range(self.seq_len) if milp_is_one(self.x[j][k])),
                        default=0,
                    ),
                    "right": max(
                        (k for k in range(self.seq_len) if milp_is_one(self.x[j][k])),
                        default=-1,
                    )
                    + 1,  # Right-side bound shall be exclusive, hence add 1
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
        fragment_predictions = pl.concat(
            [fragment_predictions, self.fragments.select(pl.col("intensity"))],
            how="horizontal",
        )

        # Reorder fragment predictions to again match the original order
        fragment_predictions = fragment_predictions.sort("orig_index")

        return seq, fragment_predictions

    def _get_sequence_nucleotide(self, k):
        for i, nuc in enumerate(self.nucleoside_names):
            if milp_is_one(self.y[i][k]):
                return nuc
        return None

    def _get_fragment_nucleotide(self, j, k):
        for i, nuc in enumerate(self.nucleoside_names):
            if milp_is_one(self.z[i][j][k]):
                return nuc
        return None
