from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import List, Self
from pulp import LpProblem, LpMinimize, LpInteger, LpContinuous, LpVariable, lpSum
from lionelmssq.alphabet import MAX_PLAUSILE_NUCLEOSIDE_DIFF, is_similar
from lionelmssq.masses import UNIQUE_MASSES
import polars as pl
from loguru import logger


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
    def __init__(self, fragments: pl.DataFrame, seq_len: int, solver: str, threads: int):
        self.fragments = fragments
        self.seq_len = seq_len
        self.solver = solver
        self.threads = threads
        self.diff_explanations = None
        self.mass_diffs = dict()
        self.singleton_masses = None

    def predict(self) -> Prediction:
        # TODO: get rid of the requirement to pass the length of the sequence
        # and instead infer it from the fragments

        self._collect_singleton_masses()
        self._collect_diffs("start")
        self._collect_diffs("end")
        self._collect_diff_explanations()

        # TODO:
        # also consider that the observations are not complete and that we probably don't see all the letters as diffs or singletons.
        # Hence, maybe do the following: solve first with the reduced alphabet, and if the optimization does not yield a sufficiently
        # good result, then try again with an extended alphabet.
        masses = self._reduce_alphabet()

        prob = LpProblem("RNA sequencing", LpMinimize)
        # i = 1,...,n: positions in the sequence
        # j = 1,...,m: fragments
        # b = 1,...,k: (modified) bases

        fragment_masses = self.fragments.get_column("observed_mass").to_list()
        start_fragments = (
            self.fragments.with_row_index()
            .filter(pl.col("is_start"))
            .get_column("index")
            .to_list()
        )
        end_fragments = (
            self.fragments.with_row_index()
            .filter(pl.col("is_end"))
            .get_column("index")
            .to_list()
        )
        n_fragments = len(fragment_masses)
        nucleosides = masses.get_column("nucleoside").to_list()
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
        # y: binary variables indicating base at position i
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
                    b: LpVariable(f"z_{i},{j},{b}", lowBound=0, upBound=1, cat=LpInteger)
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
        for j in start_fragments:
            x[0][j].setInitialValue(1)
            x[0][j].fixValue()

        # ensure that end fragments are aligned at the end of the sequence
        for j in end_fragments:
            x[self.seq_len - 1][j].setInitialValue(1)
            x[self.seq_len - 1][j].fixValue()

        # ensure that inner fragments are neither aligned at the beginning or the end of the sequence
        for j in set(range(n_fragments)) - set(start_fragments) - set(end_fragments):
            x[0][j].setInitialValue(0)
            x[0][j].fixValue()
            x[self.seq_len - 1][j].setInitialValue(0)
            x[self.seq_len - 1][j].fixValue()

        # constrain weight_diff_abs to be the absolute value of weight_diff
        for j in range(n_fragments):
            prob += predicted_mass_diff_abs[j] >= predicted_mass_diff[j]
            prob += predicted_mass_diff_abs[j] >= -predicted_mass_diff[j]

        import pulp

        match self.solver:
            case "gurobi":
                solver = "GUROBI_CMD"
            case "cbc":
                solver = "PULP_CBC_CMD"

        gurobi = pulp.getSolver(solver, threads=self.threads)
        gurobi.msg = False
        # TODO the returned value resembles the accuracy of the prediction
        _ = prob.solve(gurobi)

        def get_base(i):
            for b in nucleosides:
                if y[i][b].value() == 1:
                    return b
            return None

        # interpret solution
        seq = [get_base(i) for i in range(self.seq_len)]
        fragment_predictions = pl.from_dicts(
            [
                {
                    "left": min(i for i in range(self.seq_len) if x[i][j].value() == 1),
                    # right bound shall be exclusive, hence add 1
                    "right": max(i for i in range(self.seq_len) if x[i][j].value() == 1) + 1,
                    "observed_mass": fragment_masses[j],
                    "predicted_mass_diff": predicted_mass_diff[j].value(),
                }
                for j in range(n_fragments)
            ]
        )

        return Prediction(
            sequence=seq,
            fragments=fragment_predictions,
        )
    
    def _collect_diffs(self, side: str) -> None:
        masses = self.fragments.filter(pl.col(f"is_{side}")).get_column("observed_mass")
        self.mass_diffs[side] = [masses[0]] + (masses[1:] - masses[:-1]).to_list()
    
    def _collect_singleton_masses(self) -> None:
        masses = self.fragments.filter(~(pl.col(f"is_start") | pl.col("is_end"))).get_column("observed_mass")
        self.singleton_masses = set(mass for mass in masses if mass <= MAX_PLAUSILE_NUCLEOSIDE_DIFF)
    
    def _collect_diff_explanations(self) -> None:
        diffs = set(self.mass_diffs["start"]) | set(self.mass_diffs["end"]) | self.singleton_masses
        self.explanations = {
            diff: [
                item["nucleoside"]
                for item in UNIQUE_MASSES.iter_rows(named=True)
                if is_similar(diff, item["monoisotopic_mass"])
            ]
            for diff in diffs
        }
        # explain with two nucleosides
        self.explanations.update(
            {
                diff: [
                    (item_a["nucleoside"], item_b["nucleoside"])
                    for item_a, item_b in combinations(
                        UNIQUE_MASSES.iter_rows(named=True), 2
                    )
                    if is_similar(
                        diff, item_a["monoisotopic_mass"] + item_b["monoisotopic_mass"]
                    )
                ]
                for diff in diffs
                if not self.explanations[diff]
            }
        )

    def _reduce_alphabet(self) -> pl.DataFrame:
        def get_nucleosides():
            for expls in self.explanations.values():
                for expl in expls:
                    if isinstance(expl, tuple):
                        yield from expl
                    else:
                        yield expl
        observed_nucleosides = set(get_nucleosides())
        reduced = UNIQUE_MASSES.filter(pl.col("nucleoside").is_in(observed_nucleosides))

        return reduced