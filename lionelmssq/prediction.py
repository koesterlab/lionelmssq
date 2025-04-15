from dataclasses import dataclass
from itertools import chain, combinations, groupby
from pathlib import Path
from typing import List, Optional, Self, Set, Tuple
from pulp import LpProblem, LpMinimize, LpInteger, LpContinuous, LpVariable, lpSum
from lionelmssq.common import (
    Side,
    get_singleton_set_item,
    dag_top_n_longest_paths,
    dag_top_n_longest_paths_with_start_end,
)
from lionelmssq.masses import UNIQUE_MASSES, EXPLANATION_MASSES, MATCHING_THRESHOLD
from lionelmssq.mass_explanation import explain_mass
import polars as pl
from loguru import logger
from networkx import DiGraph, dag_longest_path
import networkx
from math import e
# from common import dag_top_n_longest_paths

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
            .sort("observed_mass")
            .with_row_index(name="index")
        )

        # Sort the fragments in the order of single nucleosides, start fragments, end fragments,
        # start fragments AND end fragments, internal fragments and then by mass for each category!

        # with pl.Config() as cfg:
        #     cfg.set_tbl_rows(-1)
        #     print(
        #         self.fragments.select(
        #             pl.col("observed_mass"),
        #             pl.col("is_start"),
        #             pl.col("is_end"),
        #             pl.col("single_nucleoside"),
        #             pl.col("is_start_end"),
        #             pl.col("is_internal"),
        #             # pl.col("mass_explanations"),
        #             pl.col("index"),
        #             pl.col("orig_index"),
        #         )
        #     )

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

    def build_skeleton(
        self,
    ):  # -> Tuple[List[Set[str]], List[TerminalFragment], List[int]]: #TODO
        skeleton_seq_start, start_fragments, invalid_start_fragments = (
            self._predict_skeleton(
                Side.START,
            )
        )

        print("Skeleton sequence start = ", skeleton_seq_start)

        skeleton_seq_end, end_fragments, invalid_end_fragments = self._predict_skeleton(
            Side.END,
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

    def graph_skeleton(
        self,
        side,
        num_top_paths=1,
        use_ms_intensity_as_weight=False,
        peanlize_explanation_length_params={"zero_len_weight": 0.0, "base": e},
        node_horizon=None,
    ):
        # Note that the penalization of explanation length can be removed by setting the base = 1

        candidate_fragments = self.fragments_side[side].get_column("index").to_list()

        masses = [0.0] + self.fragment_masses[side]

        def _accumulate_intesity(intensity, side):
            first_fragment_index = 1
            for index in range(1, len(masses) - 1):
                if abs(
                    masses[index] - masses[index + 1]
                ) < self.matching_threshold * abs(masses[index] + self.mass_tags[side]):
                    intensity[first_fragment_index] += intensity[index]
                else:
                    first_fragment_index = index + 1

        if use_ms_intensity_as_weight:
            intensity = [1.0] + self.fragments_side[side].get_column(
                "intensity"
            ).to_list()
            _accumulate_intesity(intensity, side)
        else:
            intensity = [1.0] * len(masses)
        total_intensity = sum(intensity)

        G = DiGraph()
        # Add nodes:
        nodes_list = [(i, {"mass": masses[i]}) for i in range(len(masses))]
        G.add_nodes_from(nodes_list)

        for prior_node_idx in range(len(G.nodes)):
            if node_horizon is None:
                # If the number of fragments are much larger, the number of dp calls will be very large, N(N-1)/2 calls for N fragments!
                # the node horizon can be used to calculate how many next nunmber of nodes to consider for calculating the edges!
                prior_node_horizon = len(G.nodes)
            else:
                prior_node_horizon = min(len(G.nodes), prior_node_idx + node_horizon)

            last_mass_diff = 0.0
            for latter_node_idx in range(prior_node_idx + 1, prior_node_horizon):
                mass_diff = masses[latter_node_idx] - masses[prior_node_idx]

                if (
                    abs(last_mass_diff - mass_diff)
                    < self.matching_threshold
                    * abs(masses[prior_node_idx] + self.mass_tags[side])
                ):  # NOTE: Remove this last_mass_diff and continue to include "same mass in the graph!" as well!
                    continue
                else:
                    last_mass_diff = mass_diff
                    last_node_idx = latter_node_idx
                    # The last node idx, keeps a track of the last unique "MS1" mass node,
                    # its used later to find the last node that the top paths must end at!

                if mass_diff in self.explanations:
                    mass_explanations = self.explanations.get(mass_diff, [])
                else:
                    threshold = self._calculate_diff_errors(
                        masses[prior_node_idx] + self.mass_tags[side],
                        masses[latter_node_idx] + self.mass_tags[side],
                        self.matching_threshold,
                    )
                    mass_explanations = self._calculate_diff_dp(
                        mass_diff, threshold, self.explanation_masses
                    )

                if len(mass_explanations) > 0:
                    # or abs(
                    #     mass_diff
                    # ) <= self.matching_threshold * abs(
                    #     masses[prior_node_idx] + self.mass_tags[side]
                    # ):
                    # Idea: The weight should be peanlized if the length of the explanation is more,
                    # i.e. for bigger mass differences, the weight should be less!

                    #         if len(mass_explanations) > 0:
                    #             len_explanations = min(
                    #                 len(mass_explanations[i])
                    #                 for i in range(len(mass_explanations))
                    #             )
                    #         else:
                    #             len_explanations = 0

                    len_explanations = min(
                        len(mass_explanations[i]) for i in range(len(mass_explanations))
                    )

                    G.add_edge(
                        prior_node_idx,
                        latter_node_idx,
                        weight=(
                            (
                                peanlize_explanation_length_params["zero_len_weight"]
                                + len_explanations
                            )
                            * peanlize_explanation_length_params["base"]
                            ** (-len_explanations)
                        )
                        * intensity[latter_node_idx]
                        / total_intensity,
                        explanation=mass_explanations,
                    )
                    # NOTE: If the offset is 0, it ignores the different fragments of the 'same' mass from the longest path
                    # But we add them later!
                    # To include them use a value of about 0.001 as the "zero_len_weight"!

        # longest_paths = dag_top_n_longest_paths(G, N=num_top_paths, weight="weight")
        longest_paths = dag_top_n_longest_paths_with_start_end(
            G, N=num_top_paths, weight="weight", start_node=0, end_node=last_node_idx
        )

        def _assign_valid_nodes(longest_path=longest_paths[0][1]):
            # Assign the valid and invalid nodes to the start and end fragments!
            # The valid nodes are the ones which are part of the longest path!
            # The invalid nodes are the ones which are not part of the longest path!

            valid_terminal_fragments = []
            if side == Side.START:
                pos = 0
            else:
                pos = -1

            for idx, node in enumerate(longest_path):
                if node != 0:
                    explanations = G.edges[longest_path[idx - 1], longest_path[idx]][
                        "explanation"
                    ]
                    if len(explanations) > 0:
                        if side == Side.START:
                            pos += min(
                                len(explanations[i]) for i in range(len(explanations))
                            )
                        else:
                            pos -= min(
                                len(explanations[i]) for i in range(len(explanations))
                            )
                    else:
                        pos += 0

                    valid_terminal_fragments.append(
                        TerminalFragment(
                            index=candidate_fragments[longest_path[idx] - 1],
                            min_end=pos,
                            max_end=pos,
                        )
                    )

            return valid_terminal_fragments

        def _unionize_explanations(explanations):
            len_explanations = min(
                len(explanations[i]) for i in range(len(explanations))
            )
            # TODO: Need to do this more carefully, its possible that the different explanations are of different lengths
            # One then needs to consider the min and the max positions, as Johannes did earlier!
            return [
                set(chain.from_iterable([set(expl) for expl in explanations]))
            ] * len_explanations

        def _generate_sequence_from_path(longest_path=longest_paths[0][1]):
            skeleton_seq = []
            for node in range(len(longest_path) - 1):
                if (
                    len(
                        G.edges[longest_path[node], longest_path[node + 1]][
                            "explanation"
                        ]
                    )
                    > 0
                ):
                    skeleton_seq.extend(
                        _unionize_explanations(
                            G.edges[longest_path[node], longest_path[node + 1]][
                                "explanation"
                            ]
                        )
                    )

            return skeleton_seq

        # Calculate the longest paths and the skeleton sequence
        # Remove the paths which are not of the same length as the sequence
        # Also calculate the valid terminal fragments
        skeleton_seq = []
        sequence_score = []
        valid_terminal_fragments = []
        new_longest_paths = []
        for path in longest_paths:
            seq = _generate_sequence_from_path(path[1])
            if len(seq) == self.seq_len:
                new_longest_paths.append(path)
                skeleton_seq.append(seq)
                sequence_score.append(path[0])
                valid_terminal_fragments.append(_assign_valid_nodes(path[1]))
        longest_paths = new_longest_paths

        # Add the 'same' mass fragments back to the longest paths and include them in the valid terminal fragments!
        new_longest_paths = []
        for idx_path, path in enumerate(longest_paths):

            temp_path = []
            for idx_nodes, longest_path_node in enumerate(path[1]):
                temp_path.append(longest_path_node)

                if longest_path_node != 0:
                    pos = valid_terminal_fragments[idx_path][idx_nodes - 1].min_end
                    for graph_node in G.nodes:
                        if (
                            graph_node not in path[1]
                            and graph_node not in temp_path
                            and abs(
                                G.nodes[graph_node]["mass"]
                                - G.nodes[longest_path_node]["mass"]
                            )
                            <= self.matching_threshold
                            * abs(
                                G.nodes[longest_path_node]["mass"]
                                + self.mass_tags[side]
                            )
                        ):
                            temp_path.append(graph_node)
                            # Add this node to the valid terminal fragment:
                            valid_terminal_fragments[idx_path].append(
                                TerminalFragment(
                                    index=candidate_fragments[graph_node - 1],
                                    min_end=pos,
                                    max_end=pos,
                                )
                            )

            new_longest_path = (path[0], sorted(temp_path))
            new_longest_paths.append(new_longest_path)
        longest_paths = new_longest_paths

        # Calculate the invalid nodes (not included in the longest path) for each of the longest paths
        invalid_nodes = []
        for path in longest_paths:
            invalid_nodes.append(
                [
                    candidate_fragments[node - 1]
                    for node in G.nodes
                    if node not in path[1]
                ]
            )

        for i in range(len(skeleton_seq)):
            if side == Side.END:
                skeleton_seq[i] = skeleton_seq[i][::-1]

            print(
                "Index = ",
                i,
                "Score = ",
                sequence_score[i],
                "len = ",
                len(skeleton_seq[i]),
            )
            print(f"Skeleton sequence graph {side} = ", skeleton_seq[i])

        return G, valid_terminal_fragments, skeleton_seq, invalid_nodes, sequence_score

    def predict(self) -> Prediction:
        # TODO: get rid of the requirement to pass the length of the sequence
        # and instead infer it from the fragments

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

        # Now we build the skeleton sequence from both sides and align them to get the final skeleton sequence!
        (
            _,
            start_fragments,
            skeleton_seq_start,
            invalid_start_fragments,
            seq_score_start,
        ) = self.graph_skeleton(
            Side.START,
            # use_ms_intensity_as_weight=True,
            use_ms_intensity_as_weight=False,
            num_top_paths=10,
            peanlize_explanation_length_params={"zero_len_weight": 0.0, "base": e},
        )

        # We now create reduced self.fragments_side and their masses
        # which keeps the ordereing of accepted start and end candidates while rejecting
        # the invalid ones, but keeping the ones with internal marking as internal candidates!

        _, end_fragments, skeleton_seq_end, invalid_end_fragments, seq_score_end = (
            self.graph_skeleton(
                Side.END,
                # use_ms_intensity_as_weight=True,
                use_ms_intensity_as_weight=False,
                num_top_paths=10,
                peanlize_explanation_length_params={"zero_len_weight": 0.0, "base": e},
            )
        )

        seq_set, score, start_seq_index, end_seq_index = (
            self._align_skeletons_multi_seq(
                skeleton_seq_start=skeleton_seq_start,
                skeleton_seq_end=skeleton_seq_end,
                score_seq_start=seq_score_start, #None
                score_seq_end=seq_score_end, #None
                nucleosides=nucleosides,
            )
        )

        print(seq_set)

        print("Score = ", score)

        if True:
            skeleton_seq_start = skeleton_seq_start[0]
            skeleton_seq_end = skeleton_seq_end[0]
            seq_score_start = seq_score_start[0]
            seq_score_end = seq_score_end[0]
            start_fragments = start_fragments[0]
            end_fragments = end_fragments[0]
            invalid_start_fragments = invalid_start_fragments[0]
            invalid_end_fragments = invalid_end_fragments[0]

        skeleton_seq = self._align_skeletons(
            skeleton_seq_start,
            skeleton_seq_end,
            align_depth=None,
            trust_range=None,
            trust_smaller_set=True,
        )

        print("Graph aligned_skeleton_seq = ", skeleton_seq)

        # (
        #     skeleton_seq,
        #     start_fragments,
        #     end_fragments,
        #     invalid_start_fragments,
        #     invalid_end_fragments,
        # ) = self.build_skeleton()

        # print("Ladder skeleton seq = ", skeleton_seq)

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
            self.fragments.filter(pl.col("is_internal"))
            .filter(
                ~pl.col("index").is_in(
                    [i.index for i in start_fragments]
                    + [i.index for i in end_fragments]
                )
            )
            .filter(pl.col("intensity") > 50000)
        )  # TODO: REMOVE THIS INTENSITY FILTER!!

        # TODO: One can recheck the explanations for the internal fragments, if they match with the ladder sequence. Remove the ones that do not!! Easily implemenented!

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
            j = (
                self.fragments.with_row_index("row_index")
                .filter(pl.col("index") == fragment.index)
                .select(pl.col("row_index"))
                .to_series()
                .to_list()[0]
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
                .select(pl.col("row_index"))
                .to_series()
                .to_list()[0]
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

    def _align_skeletons_multi_seq(
        self,
        skeleton_seq_start,
        skeleton_seq_end,
        score_seq_start=None,
        score_seq_end=None,
        nucleosides = {'A','U','C','G'},
    ) -> Tuple[List[List[Set[str]]], List[float], List[int], List[int]]:
        # While building the ladder it may happen that things are unambiguous from one side, but not from the other!
        # In that case, we should consider the unambiguous side as the correct one! If the intersection is empty, then we can consider the union of the two!

        # Align the skeletons of the start and end fragments to get the final skeleton sequence!
        # Wherever there is no ambiguity, that nucleotide is preferrentially considered!

        skeleton_seq = []
        skeleton_seq_score = []
        start_seq_index = []
        end_seq_index = []

        skeleton_seq_imperfect = []
        skeleton_seq_score_imperfect = []
        start_seq_index_imperfect = []
        end_seq_index_imperfect = []

        for idx_1, seq_1 in enumerate(skeleton_seq_start):
            for idx_2, seq_2 in enumerate(skeleton_seq_end):
                temp_seq = [set() for _ in range(self.seq_len)]
                perfect_match = True
                temp_score = 0.

                for i in range(self.seq_len):
                    temp_seq[i] = seq_1[i].intersection(seq_2[i])
                    if temp_seq[i]:
                        if score_seq_start is not None and score_seq_end is not None:
                            # temp_score += float(len(temp_seq[i]))/(
                            #     score_seq_start[idx_1] + score_seq_end[idx_2]
                            # )*float(min(i,abs(i-self.seq_len)))/self.seq_len
                            temp_score += float(len(nucleosides) - len(temp_seq[i]))*(
                                score_seq_start[idx_1] + score_seq_end[idx_2]
                            )
                        else:
                            # temp_score += float(len(temp_seq[i]))*float(min(i,abs(i-self.seq_len)))/self.seq_len
                            temp_score += float(len(nucleosides) - len(temp_seq[i]))
                    else:
                        if score_seq_start is not None and score_seq_end is not None:
                            temp_score += float(len(nucleosides))/(
                                score_seq_start[idx_1] + score_seq_end[idx_2]
                            ) #Penalize this by the number of nucelosides being considered!
                        else:
                            temp_score += float(len(nucleosides)) #Penalize this by the number of nucelosides being considered!

                        if len(seq_1[i]) < len(seq_2[i]):
                            temp_seq[i] = seq_1[i]
                        else:
                            temp_seq[i] = seq_2[i]
                        perfect_match = False

                if perfect_match and temp_seq not in skeleton_seq:
                    skeleton_seq.append(temp_seq)
                    skeleton_seq_score.append(temp_score)
                    start_seq_index.append(idx_1)
                    end_seq_index.append(idx_2)

                if not perfect_match and temp_seq not in skeleton_seq:
                    skeleton_seq_imperfect.append(temp_seq)
                    skeleton_seq_score_imperfect.append(temp_score)
                    start_seq_index_imperfect.append(idx_1)
                    end_seq_index_imperfect.append(idx_2)

        if skeleton_seq:
            sorted_skeleton_seq = [
                seq
                for _, seq in sorted(zip(skeleton_seq_score, skeleton_seq), reverse=True)
            ]
            sorted_start_seq_index = [
                index
                for _, index in sorted(
                    zip(skeleton_seq_score, start_seq_index), reverse=True
                )
            ]
            sorted_end_seq_index = [
                index
                for _, index in sorted(
                    zip(skeleton_seq_score, end_seq_index), reverse=True
                )
            ]
            sorted_skeleton_seq_score = sorted(skeleton_seq_score, reverse=True)

        elif skeleton_seq_imperfect:
            sorted_skeleton_seq = [
                seq
                for _, seq in sorted(
                    zip(skeleton_seq_score_imperfect, skeleton_seq_imperfect),
                    reverse=True,
                )
            ]
            sorted_start_seq_index = [
                index
                for _, index in sorted(
                    zip(skeleton_seq_score_imperfect, start_seq_index_imperfect),
                    reverse=True,
                )
            ]
            sorted_end_seq_index = [
                index
                for _, index in sorted(
                    zip(skeleton_seq_score_imperfect, end_seq_index_imperfect),
                    reverse=True,
                )
            ]
            sorted_skeleton_seq_score = sorted(skeleton_seq_score_imperfect, reverse=True)

        return (
            sorted_skeleton_seq,
            sorted_skeleton_seq_score,
            sorted_start_seq_index,
            sorted_end_seq_index,
        )

    def _align_skeletons(
        self,
        skeleton_seq_start,
        skeleton_seq_end,
        align_depth=None,
        trust_range=None,
        trust_smaller_set=False,
    ) -> List[Set[str]]:
        # While building the ladder it may happen that things are unambiguous from one side, but not from the other!
        # In that case, we should consider the unambiguous side as the correct one! If the intersection is empty, then we can consider the union of the two!

        # Align the skeletons of the start and end fragments to get the final skeleton sequence!
        # Wherever there is no ambiguity, that nucleotide is preferrentially considered!

        skeleton_seq = [set() for _ in range(self.seq_len)]

        if align_depth is None:
            if trust_range is None:
                for i in range(self.seq_len):
                    skeleton_seq[i] = skeleton_seq_start[i].intersection(
                        skeleton_seq_end[i]
                    )
                    if not skeleton_seq[
                        i
                    ]:  # TODO: If the intersection is empty, then we can also choose to trust the left sequence in the left part and the right sequence in the right part!
                        if not trust_smaller_set:
                            skeleton_seq[i] = skeleton_seq_start[i].union(
                                skeleton_seq_end[i]
                            )
                        else:
                            if len(skeleton_seq_start[i]) < len(skeleton_seq_end[i]):
                                skeleton_seq[i] = skeleton_seq_start[i]
                            else:
                                skeleton_seq[i] = skeleton_seq_end[i]
            else:
                trust_range = int(self.seq_len / 2)  # TODO: update this algo smartly!
                for i in range(trust_range):
                    skeleton_seq[i] = skeleton_seq_start[i]
                for i in range(trust_range, self.seq_len):
                    skeleton_seq[i] = skeleton_seq_end[i]
        else:
            for i in range(align_depth):
                skeleton_seq[i] = skeleton_seq_start[i].intersection(
                    skeleton_seq_end[i]
                )
                if not skeleton_seq[i]:
                    skeleton_seq[i] = skeleton_seq_start[i].union(skeleton_seq_end[i])
            # for i in range(align_depth, self.seq_len):
            for i in range(-align_depth, 0):
                skeleton_seq[i] = skeleton_seq_start[i].intersection(
                    skeleton_seq_end[i]
                )
                if not skeleton_seq[i]:
                    skeleton_seq[i] = skeleton_seq_start[i].union(skeleton_seq_end[i])

        # TODO: Its more complicated, since if two positions are ambigious, they are not indepenedent.
        # If one nucleotide is selected this way, then the same nucleotide cannot be selected in the other position!

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
                0 <= pos <= self.seq_len
                if side == Side.START
                else -(self.seq_len + 1) <= pos < 0
                # else -self.seq_len <= pos < 0
            )

        pos = {0} if side == Side.START else {-1}
        carry_over_mass = 0

        valid_terminal_fragments = []
        invalid_fragments = []
        last_mass_valid = 0.0
        for fragment_index, (diff, mass) in enumerate(
            zip(self.mass_diffs[side], fragment_masses)
        ):
            diff += carry_over_mass
            assert pos

            is_valid = True
            if diff in self.explanations:
                explanations = self.explanations.get(diff, [])
                # last_mass_valid = mass
            else:
                # CHECK: Review if the correct values of the mass, i.e "last_mass_valid" are used here!
                threshold = self._calculate_diff_errors(
                    last_mass_valid + self.mass_tags[side],
                    last_mass_valid + diff + self.mass_tags[side],
                    self.matching_threshold,
                )
                explanations = self._calculate_diff_dp(
                    diff, threshold, self.explanation_masses
                )
            # if explanations:
            #     carry_over_mass = 0.0 #CHECK: Review this!

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
                            # print("i = ", i)
                            # print("possible_nucleosides = ", possible_nucleosides)
                            # print("alphabet = ", alphabet)
                            # TODO: Review this! What's the need to clear it really?
                            # if possible_nucleosides.issuperset(alphabet): #Should this be the other way around?
                            #     # the expl sharpens the possibilities
                            #     # clear the possibilities so far, the expl will add
                            #     # the sharpened ones ones below
                            #     possible_nucleosides.clear()

                            for j in alphabet:
                                possible_nucleosides.add(j)
                                # TODO: We need to do this better.
                                # Instead of adding just the letters, we somehow need to keep a track of the possibilities to be able to constrain the LP!
                                # We also then probably need part of the code immediately below!

                    # for expl in p_specific_explanations:
                    #         #Maybe we can save the object as perm itself and block it for sometime!
                    #         for perm in permutations(expl):
                    #             for i, nuc in enumerate(perm):
                    #                 get_possible_nucleosides(p, i).add(nuc)

                    # print("Intermediate skeletal seq = ", skeleton_seq)

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
                # abs(diff/mass) <= self._calculate_diff_errors(mass+self.mass_tags[side],mass+diff+self.mass_tags[side],self.matching_threshold)
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
                        # index=fragment_index,
                        min_end=min_fragment_end,
                        max_end=max_fragment_end,
                    )
                )
                pos = next_pos
                last_mass_valid = mass  # CHECK:Review this!
                carry_over_mass = 0.0  # TODO:Review this!
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
        return f"{{{",".join(self.nucleosides)}}}"
