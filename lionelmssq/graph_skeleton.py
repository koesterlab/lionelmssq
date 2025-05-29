from math import e
from networkx import DiGraph, topological_sort
from heapq import heappush, heappop
from itertools import chain
from lionelmssq.common import Side
from lionelmssq.common import TerminalFragment
from copy import deepcopy


def construct_graph_skeleton(
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
            if abs(masses[index] - masses[index + 1]) < self.matching_threshold * abs(
                masses[index] + self.mass_tags[side]
            ):
                intensity[first_fragment_index] += intensity[index]
            else:
                first_fragment_index = index + 1

    if use_ms_intensity_as_weight:
        intensity = [1.0] + self.fragments_side[side].get_column("intensity").to_list()
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
        pos = 0
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
                #TODO: Check out the use of this min, for multiple explanations with differernt lengths, 
                #this will be problematic!

                pos += len_explanations

                G.add_edge(
                    prior_node_idx,
                    latter_node_idx,
                    weight=(
                        (
                            (
                                peanlize_explanation_length_params["zero_len_weight"]
                                + len_explanations
                            )
                            * peanlize_explanation_length_params["base"]
                            ** (-len_explanations)
                        )
                        + e ** (-pos / self.seq_len)
                    )  # Score the sequences with shorter explanations higher when they are at the beginning of the sequence!
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
        print("Explanations = ", explanations)
        len_explanations = min(len(explanations[i]) for i in range(len(explanations)))
        # TODO: Need to do this more carefully, its possible that the different explanations are of different lengths
        # One then needs to consider the min and the max positions, as Johannes did earlier!
        return [
            set(chain.from_iterable([set(expl) for expl in explanations]))
        ] * len_explanations

    def _generate_sequence_from_path(longest_path=longest_paths[0][1]):
        skeleton_seq = []
        for node in range(len(longest_path) - 1):
            if (
                len(G.edges[longest_path[node], longest_path[node + 1]]["explanation"])
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

    # def _generate_list_explanations(longest_path=longest_paths[0][1]):
    #     skeleton_seq = []
    #     for node in range(len(longest_path) - 1):
    #         if (
    #             len(
    #                 G.edges[longest_path[node], longest_path[node + 1]][
    #                     "explanation"
    #                 ]
    #             )
    #             > 0
    #         ):
    #             seq = G.edges[longest_path[node], longest_path[node + 1]][
    #                         "explanation"
    #                     ]
    #             list_seq = [[str(s) for s in subseq] for subseq in seq]

    #             skeleton_seq.extend(
    #                     list_seq
    #                 )
    #     return skeleton_seq

    def _generate_list_explanations_all_combinations(longest_path=longest_paths[0][1]):
        skeleton_seq = []
        for node in range(len(longest_path) - 1):
            if (
                len(G.edges[longest_path[node], longest_path[node + 1]]["explanation"])
                > 0
            ):
                seq = G.edges[longest_path[node], longest_path[node + 1]]["explanation"]

                list_seq = [[str(s) for s in subseq] for subseq in seq]

                len_list_seq = len(list_seq)

                if not skeleton_seq:
                    for subseq in list_seq:
                        skeleton_seq.extend([[subseq]])
                else:
                    original_skeleton_seq = deepcopy(skeleton_seq)
                    for skeleton in skeleton_seq:
                        skeleton.extend([list_seq[0]])

                    if len(list_seq) > 1:
                        for skeleton in original_skeleton_seq:
                            skeleton_seq.extend(
                                skeleton + [list_seq[i]] for i in range(1, len_list_seq)
                            )

        return skeleton_seq

    # Calculate the longest paths and the skeleton sequence
    # Remove the paths which are not of the same length as the sequence
    # Also calculate the valid terminal fragments
    skeleton_seq = []
    sequence_score = []
    valid_terminal_fragments = []
    new_longest_paths = []
    list_explanations = []
    for path in longest_paths:
        seq = _generate_sequence_from_path(path[1])
        # if len(seq) == self.seq_len: #TODO: CHECK
        #     new_longest_paths.append(path)
        #     skeleton_seq.append(seq)
        #     sequence_score.append(path[0])
        #     valid_terminal_fragments.append(_assign_valid_nodes(path[1]))
        #     print("Path = ", path[1])
        #     print("Mass = ", [G.nodes[node]["mass"] for node in path[1]])
        #     print("Score = ", path[0])

        list_explanations_single = _generate_list_explanations_all_combinations(path[1])
        for exp in list_explanations_single: 
            if sum(len(sublist) for sublist in exp) != self.seq_len:
                list_explanations_single.remove(exp)

        # list_explanations.append(list_explanations_single)

        #TODO: CHECK
        if list_explanations_single:
            new_longest_paths.append(path)
            skeleton_seq.append(seq)
            sequence_score.append(path[0])
            valid_terminal_fragments.append(_assign_valid_nodes(path[1]))
            print("Path = ", path[1])
            print("Mass = ", [G.nodes[node]["mass"] for node in path[1]])
            print("Score = ", path[0])
            list_explanations.append(list_explanations_single)

            if side == Side.END:
                print("Seq = ", seq[::-1])
                print("List_explnations = ", list_explanations_single[:][::-1])
            else:
                print("Seq = ", seq)
                print("List_explnations = ", list_explanations_single)

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
                        * abs(G.nodes[longest_path_node]["mass"] + self.mass_tags[side])
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
            [candidate_fragments[node - 1] for node in G.nodes if node not in path[1]]
        )

    for i in range(len(skeleton_seq)):
        if side == Side.END:
            skeleton_seq[i] = skeleton_seq[i][::-1]
            list_explanations[i] = list_explanations[i][::-1]

        # print(
        #     "Index = ",
        #     i,
        #     "Score = ",
        #     sequence_score[i],
        #     "len = ",
        #     len(skeleton_seq[i]),
        # )
        # print(f"Skeleton sequence graph {side} = ", skeleton_seq[i])

    return (
        G,
        valid_terminal_fragments,
        skeleton_seq,
        invalid_nodes,
        sequence_score,
        list_explanations,
    )


def dag_top_n_longest_paths(G, N, weight="weight", default_weight=1, topo_order=None):
    """Returns the top N longest paths in a directed acyclic graph (DAG).
    This function extends the networkx function "dag_longest_path" to output the top N paths

    Parameters
    ----------
    G : NetworkX DiGraph
        A directed acyclic graph (DAG)

    N : int
        The number of longest paths to return

    weight : str, optional
        Edge data key to use for weight

    default_weight : int, optional
        The weight of edges that do not have a weight attribute

    topo_order: list or tuple, optional
        A topological order for `G` (if None, the function will compute one)

    Returns
    -------
    list of tuples
        A list of the top N longest paths, where each path is represented as a tuple
        (path_length, path_nodes)
    """
    if not G:
        return []

    if topo_order is None:
        topo_order = list(topological_sort(G))

    # Dictionary to store the top N longest paths to each node
    dist = {v: [] for v in G.nodes}

    for v in topo_order:
        # Get all predecessors of the current node
        for u, data in G.pred[v].items():
            # Calculate the weight of the edge
            edge_weight = data.get(weight, default_weight)

            # Update the top N paths for the current node
            for path_length, path_nodes in dist[u]:
                new_path_length = path_length + edge_weight
                new_path_nodes = path_nodes + [v]

                # Add the new path to the priority queue for node `v`
                heappush(dist[v], (new_path_length, new_path_nodes))

                # Keep only the top N longest paths
                if len(dist[v]) > N:
                    heappop(dist[v])

        # If no predecessors, initialize the node with a single path
        if not dist[v]:
            heappush(dist[v], (0, [v]))

    # Collect all paths from all nodes and find the top N longest paths overall
    all_paths = list(chain.from_iterable(dist.values()))
    all_paths.sort(reverse=True, key=lambda x: x[0])  # Sort by path length (descending)

    return all_paths[:N]


def dag_top_n_longest_paths_with_start_end(
    G, N, start_node, end_node, weight="weight", default_weight=1
):
    """Returns the top N longest paths in a DAG that start at a specific node and end at a specific node.

    Parameters
    ----------
    G : NetworkX DiGraph
        A directed acyclic graph (DAG)

    N : int
        The number of longest paths to return

    start_node : node
        The node where the paths must start

    end_node : node
        The node where the paths must end

    weight : str, optional
        Edge data key to use for weight

    default_weight : int, optional
        The weight of edges that do not have a weight attribute

    Returns
    -------
    list of tuples
        A list of the top N longest paths, where each path is represented as a tuple
        (path_length, path_nodes)
    """
    if not G:
        return []

    # Ensure start_node and end_node exist in the graph
    if start_node not in G or end_node not in G:
        raise ValueError(
            f"Start node {start_node} or end node {end_node} is not in the graph."
        )

    # Compute topological order
    topo_order = list(topological_sort(G))

    # Initialize DP table: {node: priority queue of top N paths}
    dp = {v: [] for v in G.nodes}

    # Initialize the start node with a single path of length 0
    heappush(dp[start_node], (0, [start_node]))

    # Process nodes in topological order
    for v in topo_order:
        # # Skip nodes that have no paths leading to them
        # if not dp[v]:
        #     continue

        # Update paths for the current node based on its predecessors
        for u, data in G.pred[v].items():
            edge_weight = data.get(weight, default_weight)

            # Add paths from predecessor `u` to the current node `v`
            for path_length, path_nodes in dp[u]:
                new_path_length = path_length + edge_weight
                new_path_nodes = path_nodes + [v]

                # Add the new path to the priority queue for node `v`
                heappush(dp[v], (new_path_length, new_path_nodes))

                # Keep only the top N longest paths
                if len(dp[v]) > N:
                    heappop(dp[v])

    # Extract the top N longest paths that end at the specified end_node
    all_paths = dp[end_node]
    all_paths.sort(reverse=True, key=lambda x: x[0])  # Sort by path length (descending)

    print("All paths = ", all_paths)

    return all_paths[:N]
