# -*- coding: utf-8 -*-
"""Module for sequence inference."""

from itertools import combinations
from typing import Any, Set

import numpy as np
import polars as pl
from pulp import (
    LpContinuous,
    LpInteger,
    LpMinimize,
    LpProblem,
    LpVariable,
    getSolver,
    lpSum,
)

from spectrseqtools.dataclasses import (
    PredictedFragments,
    Prediction,
    Sequence,
    SequenceInformation,
    SolverParameters,
)
from spectrseqtools.nucleotide_alphabet import NucleotideAlphabet
from spectrseqtools.sequence import SkeletonSequence

MILP_QUASI_ONE_THRESHOLD = 0.9


def milp_is_one(var, threshold=MILP_QUASI_ONE_THRESHOLD):
    """Return whether variable is over threshold."""
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
    """Class for linear program instance."""

    def __init__(
        self,
        fragments,
        alphabet: NucleotideAlphabet,
        seq: SequenceInformation,
        skeleton_seq: SkeletonSequence,
    ):
        # i = 1,...,N: (modified) nucleotides
        # j = 1,...,M: fragments
        # k = 1,...,S: positions in the sequence
        self.fragments = fragments
        self.seq = seq
        self.alphabet = alphabet

        # x: binary variables indicating fragment j presence at position k
        self.x = self._set_x()
        # y: binary variables indicating nucleotide i at position k
        self.y = self._set_y(skeleton_seq)
        # z: binary variables indicating product of x and y
        self.z = self._set_z()

        # predicted_mass_diff: difference between a fragment's SU-mass
        # and the sum of nucleotide masses assigned by the MILP prediction
        self.predicted_mass_diff = self._set_predicted_mass_difference()

    def _set_x(self):
        """Return binary variables indicating fragment j presence at position k."""
        x = [
            [
                LpVariable(f"x_{j},{k}", lowBound=0, upBound=1, cat=LpInteger)
                for k in range(self.seq.max_len)
            ]
            for j in range(len(self.fragments))
        ]

        for j in range(len(self.fragments)):
            # Ensure intact fragments are aligned at the whole sequence
            if self.fragments.item(j, "fragmentation") == "START_END":
                for k in range(self.seq.max_len):
                    x[j][k].setInitialValue(1)
                    x[j][k].fixValue()
                continue

            # Ensure START fragments are aligned at the beginning of the sequence
            if "START" in self.fragments.item(j, "fragmentation"):
                # min_end is exclusive
                for k in range(self.fragments.item(j, "min_end") + 1):
                    x[j][k].setInitialValue(1)
                    x[j][k].fixValue()
                for k in range(self.fragments.item(j, "max_end") + 1, self.seq.max_len):
                    x[j][k].setInitialValue(0)
                    x[j][k].fixValue()
                continue

            # Ensure END fragments are aligned at the end of the sequence
            if "END" in self.fragments.item(j, "fragmentation"):
                # min_end is exclusive
                for k in range(self.fragments.item(j, "max_end")):
                    x[j][k].setInitialValue(0)
                    x[j][k].fixValue()
                for k in range(self.fragments.item(j, "min_end"), self.seq.max_len):
                    x[j][k].setInitialValue(1)
                    x[j][k].fixValue()
                continue

            # Internal fragments are not further constrained in both their
            # positioning and length for now; let the LP decide.

        return x

    def _set_y(self, skeleton_seq):
        """Return binary variables indicating nucleotide i at position k."""
        y = [
            [
                LpVariable(f"y_{i},{k}", lowBound=0, upBound=1, cat=LpInteger)
                for k in range(self.seq.max_len)
            ]
            for i in range(len(self.alphabet))
        ]

        # Use skeleton sequence to fix nucleotides
        for k, nucs in enumerate(skeleton_seq):
            if not nucs:
                # Do not constrain if nothing is known
                continue
            for i in range(len(self.alphabet)):
                # Do not allow nucleotides that are not observed in the skeleton
                if i not in nucs:
                    y[i][k].setInitialValue(0)
                    y[i][k].fixValue()
                # If only one nucleotide is possible, fix the value already
                if len(nucs) == 1:
                    if i == get_singleton_set_item(nucs):
                        y[i][k].setInitialValue(1)
                        y[i][k].fixValue()

        return y

    def _set_z(self):
        """Return binary variables indicating product of x and y."""
        z = [
            [
                [
                    LpVariable(f"z_{i},{j},{k}", lowBound=0, upBound=1, cat=LpInteger)
                    for k in range(self.seq.max_len)
                ]
                for j in range(len(self.fragments))
            ]
            for i in range(len(self.alphabet))
        ]
        return z

    def _set_predicted_mass_difference(self):
        """Return variables indicating predicted mass difference."""
        fragment_masses = self.fragments.get_column("standard_unit_mass").to_list()
        return [
            fragment_masses[j]
            - lpSum(
                [
                    self.z[i][j][k] * self.alphabet.get(i).nucleotide_mass
                    for i in range(len(self.alphabet))
                    for k in range(self.seq.max_len)
                ]
            )
            for j in range(len(self.fragments))
        ]

    def _define_lp_problem(self, full_problem: bool = True):
        """Return linear-problem instance."""
        problem = LpProblem("fragment_filter", LpMinimize)

        # predicted_mass_diff_abs: absolute value of predicted_mass_diff
        predicted_mass_diff_abs = [
            LpVariable(f"predicted_mass_diff_abs_{j}", lowBound=0, cat=LpContinuous)
            for j in range(len(self.fragments))
        ]

        # Set optimization function
        problem += lpSum(
            [predicted_mass_diff_abs[j] for j in range(len(self.fragments))]
        )

        # Select one nucleotide per position
        for k in range(self.seq.max_len):
            problem += lpSum([self.y[i][k] for i in range(len(self.alphabet))]) == 1

        # Enforce universal modification rate
        problem += (
            lpSum(
                [
                    self.y[i][k]
                    for k in range(self.seq.max_len)
                    for i in range(len(self.alphabet))
                    if self.alphabet.get(i).is_modification
                ]
            )
            <= self.seq.max_modifications
        )

        # Enforce individual modification rates
        for i, mass in enumerate(self.alphabet.alphabet):
            problem += lpSum(
                [self.y[i][k] for k in range(self.seq.max_len)]
            ) <= np.ceil(mass.modification_rate * self.seq.max_len)

        # Fill z with the product of binary variables x and y
        for k in range(self.seq.max_len):
            for j in range(len(self.fragments)):
                for i in range(len(self.alphabet)):
                    problem += self.z[i][j][k] <= self.x[j][k]
                    problem += self.z[i][j][k] <= self.y[i][k]
                    problem += self.z[i][j][k] >= self.x[j][k] + self.y[i][k] - 1

        # Ensure that fragments are aligned continuously (i.e. no gaps:
        # if x[j, k1] = 1 and x[j, k2] = 1, then x[j, k_between] = 1)
        for j in range(len(self.fragments)):
            for k1, k2 in combinations(range(self.seq.max_len), 2):
                # k1 and k2 are inclusive
                assert k2 > k1
                if k2 - k1 > 1:
                    problem += (self.x[j][k1] + self.x[j][k2] - 1) * (
                        k2 - k1 - 1
                    ) <= lpSum(
                        [self.x[j][k_between] for k_between in range(k1 + 1, k2)]
                    )

        # Ensure that predicted sequence matches intact mass (for full LP)
        if full_problem:
            problem += (
                lpSum(
                    [
                        self.y[i][k] * self.alphabet.get(i).nucleotide_mass
                        for k in range(self.seq.max_len)
                        for i in range(len(self.alphabet))
                    ]
                )
                >= self.seq.su_mass - self.seq.max_variance
            )
            problem += self.seq.su_mass + self.seq.max_variance >= lpSum(
                [
                    self.y[i][k] * self.alphabet.get(i).nucleotide_mass
                    for k in range(self.seq.max_len)
                    for i in range(len(self.alphabet))
                ]
            )

        # Constrain predicted_mass_diff_abs to be the absolute value of predicted_mass_diff
        for j in range(len(self.fragments)):
            problem += predicted_mass_diff_abs[j] >= self.predicted_mass_diff[j]
            problem += predicted_mass_diff_abs[j] >= -1 * self.predicted_mass_diff[j]

        return problem

    def minimize_error(self, solver_params: SolverParameters) -> float:
        """
        Return minimal error.

        Parameters
        ----------
        solver_params : SolverParameters
            Solver parameters.

        Returns
        -------
        float
            Minimal error within timeframe.

        """
        try:
            # Initialize solver
            solver = getSolver(**solver_params.to_dict(filter_only=True))

            lp_problem = self._define_lp_problem(full_problem=False)
            _ = lp_problem.solve(solver)
            score = lp_problem.objective.value()
            return np.inf if score is None else score
        except Exception:
            return np.inf

    def evaluate(self, solver_params: SolverParameters) -> Prediction:
        """
        Evaluate prediction.

        Parameters
        ----------
        solver_params : SolverParameters
            Solver parameters.

        Returns
        -------
        Prediction
            Best prediction result within timeframe.

        """
        # Initialize solver
        solver = getSolver(**solver_params.to_dict(filter_only=False))

        # TODO: Make returned value resemble prediction accuracy
        lp_problem = self._define_lp_problem()
        _ = lp_problem.solve(solver)
        print(f"LP status after solving: {lp_problem.status}\n")

        # Interpret solution
        return Prediction(
            sequence=self._get_sequence(), fragments=self._get_fragments()
        )

    def _get_sequence(self) -> Sequence:
        return Sequence(
            sequence=[
                self.alphabet.get(self._get_sequence_nucleotide(k)).representative
                for k in range(self.seq.max_len)
            ]
        )

    def _get_sequence_nucleotide(self, k):
        for i in range(len(self.alphabet)):
            if milp_is_one(self.y[i][k]):
                return i
        return None

    def _get_fragments(self) -> PredictedFragments:
        fragment_masses = self.fragments.get_column("standard_unit_mass").to_list()

        # Get the sequence corresponding to each of the fragments!
        fragment_seq = [
            "".join(
                [
                    self.alphabet.get(
                        self._get_fragment_nucleotide(j, k)
                    ).representative
                    for k in range(self.seq.max_len)
                    if self._get_fragment_nucleotide(j, k) is not None
                ]
            )
            for j in list(range(len(fragment_masses)))
        ]

        # Get the mass corresponding to each of the fragments!
        predicted_fragment_mass = [
            sum(
                (
                    self.alphabet.get(
                        self._get_fragment_nucleotide(j, k)
                    ).nucleotide_mass
                    for k in range(self.seq.max_len)
                    if self._get_fragment_nucleotide(j, k) is not None
                )
            )
            for j in list(range(len(fragment_masses)))
        ]

        observed_masses = self.fragments.get_column("observed_mass").to_list()
        fragment_predictions = pl.from_dicts(
            [
                {
                    # Because of the relaxation of the LP, sometimes the value is not exactly 1
                    "left": self._get_leftmost_position(j),
                    "right": self._get_rightmost_position(j),
                    "observed_mass": observed_masses[j],
                    "standard_unit_mass": round(
                        fragment_masses[j], self.alphabet.decimal_places
                    ),
                    "predicted_mass": round(
                        predicted_fragment_mass[j], self.alphabet.decimal_places
                    ),
                    "predicted_diff": round(
                        self.predicted_mass_diff[j].value(),
                        self.alphabet.decimal_places,
                    ),
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
        return PredictedFragments(fragments=fragment_predictions.sort("orig_index"))

    def _get_fragment_nucleotide(self, j, k):
        for i in range(len(self.alphabet)):
            if milp_is_one(self.z[i][j][k]):
                return i
        return None

    def _get_leftmost_position(self, j):
        return min(
            (k for k in range(self.seq.max_len) if milp_is_one(self.x[j][k])),
            default=0,
        )

    def _get_rightmost_position(self, j):
        return (
            max(
                (k for k in range(self.seq.max_len) if milp_is_one(self.x[j][k])),
                default=-1,
            )
            + 1  # Right-side bound shall be exclusive, hence add 1
        )
