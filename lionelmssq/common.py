from enum import Enum
import re
from typing import Any, Set
from heapq import heappush, heappop
from itertools import chain
import networkx as nx


# _NUCLEOSIDE_RE = re.compile(r"\d*[ACGUT]")
_NUCLEOSIDE_RE = re.compile(r"\d*[ACGU]")


def parse_nucleosides(sequence: str):
    return _NUCLEOSIDE_RE.findall(sequence)


class Side(Enum):
    START = "start"
    END = "end"

    def __str__(self):
        return self.value


def get_singleton_set_item(set_: Set[Any]) -> Any:
    """Return the only item in a set."""
    if len(set_) != 1:
        raise ValueError(f"Expected a set with one item, got {set_}")
    return next(iter(set_))


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
        topo_order = list(nx.topological_sort(G))

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
    topo_order = list(nx.topological_sort(G))

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

    return all_paths[:N]
