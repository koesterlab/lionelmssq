import re
from typing import Any, List, Set

from lionelmssq.mass_explanation import explain_mass_with_table
from lionelmssq.mass_table import DynamicProgrammingTable

MILP_QUASI_ONE_THRESHOLD = 0.9


# _NUCLEOSIDE_RE = re.compile(r"\d*[ACGUT]")
_NUCLEOSIDE_RE = re.compile(r"\d*[ACGU]")


def milp_is_one(var, threshold=MILP_QUASI_ONE_THRESHOLD):
    # Sometime the LP does not exactly output probabilities of 1 for one nucleotide or one position.
    # This is due to the LP relaxation. Hence, we need to set a threshold for the LP relaxation.
    return var.value() >= threshold


def parse_nucleosides(sequence: str):
    return _NUCLEOSIDE_RE.findall(sequence)


def get_singleton_set_item(set_: Set[Any]) -> Any:
    """Return the only item in a set."""
    if len(set_) != 1:
        raise ValueError(f"Expected a set with one item, got {set_}")
    return next(iter(set_))


class Explanation:
    def __init__(self, *nucleosides):
        self.nucleosides = tuple(sorted(nucleosides))

    def __iter__(self):
        yield from self.nucleosides

    def __len__(self):
        return len(self.nucleosides)

    def __repr__(self):
        return f"{{{','.join(self.nucleosides)}}}"


def calculate_error_threshold(mass1: float, mass2: float, threshold: float) -> float:
    # METHOD: Determine the error threshold using the L2 norm
    return threshold * ((mass1**2 + mass2**2) ** 0.5)


def calculate_explanations(
    diff: float,
    threshold: float,
    modification_rate: float,
    seq_len: int,
    dp_table: DynamicProgrammingTable,
) -> List[Explanation]:
    explanation_list = explain_mass_with_table(
        diff,
        dp_table=dp_table,
        seq_len=seq_len,
        max_modifications=round(modification_rate * seq_len),
        threshold=threshold,
    ).explanations

    # Return None if no explanation was found
    if explanation_list is None:
        return None

    # Return all found explanations
    explanation_list = list(explanation_list)
    return [Explanation(*explanation_list[i]) for i in range(len(explanation_list))]
