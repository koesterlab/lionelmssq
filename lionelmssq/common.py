import re
from typing import List

from lionelmssq.mass_explanation import explain_mass_with_table
from lionelmssq.mass_table import DynamicProgrammingTable

ERROR_METHOD = "l2_norm"
_NUCLEOSIDE_RE = re.compile(r"\d*[ACGU]")


def parse_nucleosides(sequence: str):
    return _NUCLEOSIDE_RE.findall(sequence)


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
    match ERROR_METHOD:
        case "l1_norm":
            return threshold * (mass1 + mass2)
        case "l2_norm":
            return threshold * ((mass1**2 + mass2**2) ** 0.5)
        case _:
            raise NotImplementedError("This error method is not implemented.")


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
