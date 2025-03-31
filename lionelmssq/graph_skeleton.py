from dataclasses import dataclass
from itertools import chain, combinations, groupby
from pathlib import Path
from typing import List, Optional, Self, Set, Tuple
from pulp import LpProblem, LpMinimize, LpInteger, LpContinuous, LpVariable, lpSum
from lionelmssq.common import Side, get_singleton_set_item
from lionelmssq.masses import UNIQUE_MASSES, EXPLANATION_MASSES, MATCHING_THRESHOLD
from lionelmssq.mass_explanation import explain_mass
import polars as pl
from loguru import logger

from networkx import DiGraph, dag_longest_path
# import networkx as nx


def graph_skeleton(side, node_horizon=None):
    G = DiGraph()

    # First assume that there are no almost identical masses, we will group them from the table later!
    masses = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    multiplicity = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    # Add nodes:
    G.add_node(0, mass=0, weight=1)  # Source node
    G.add_node(
        range(1, len(masses)), mass=masses, weight=multiplicity
    )  # Nodes for all masses

    for prior_node_idx in range(len(G.nodes)):
        if node_horizon is None:
            prior_node_horizon = len(G.nodes)
        else:
            prior_node_horizon = min(len(G.nodes), prior_node_idx + node_horizon)

        for latter_node_idx in range(prior_node_idx + 1, prior_node_horizon):
            mass_diff = masses[latter_node_idx] - masses[prior_node_idx]
            # Need to explain this mass with the proper threshold!
            mass_explanations = list(
                explain_mass(
                    mass_diff, MATCHING_THRESHOLD, EXPLANATION_MASSES
                ).explanations
            )

            if (
                len(mass_explanations) > 0 | mass_diff < 0.0001
            ):  # introduce a threshold for mass difference!
                # G.add_edge(prior_node_idx,latter_node_idx,multiplicity=multiplicity[latter_node_idx],explanation=mass_explanations)
                G.add_edge(
                    prior_node_idx,
                    latter_node_idx,
                    weight=G.nodes[latter_node_idx]["multiplicity"],
                    explanation=mass_explanations,
                )

    # Start from node 0, calculate difference to all nodes in the horizon
    # Then from node 1 and so on, create a graph with directed edges and multiplicity of nodes as weights

    return G, dag_longest_path(G, weight="weight")


def longest_path(G, source, sink):
    # https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.simple_paths.longest_simple_path.html#networkx.algorithms.simple_paths.longest_simple_path
    return nx.algorithms.simple_paths.longest_simple_path(G, source, sink)
