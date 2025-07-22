from enum import Enum
import re
from typing import Any, Set

from lionelmssq.mass_explanation import explain_mass

MILP_QUASI_ONE_THRESHOLD = 0.9


# _NUCLEOSIDE_RE = re.compile(r"\d*[ACGUT]")
_NUCLEOSIDE_RE = re.compile(r"\d*[ACGU]")


def milp_is_one(var, threshold=MILP_QUASI_ONE_THRESHOLD):
    # Sometime the LP does not exactly output probabilities of 1 for one nucleotide or one position.
    # This is due to the LP relaxation. Hence, we need to set a threshold for the LP relaxation.
    return var.value() >= threshold


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


class Explanation:
    def __init__(self, *nucleosides):
        self.nucleosides = tuple(sorted(nucleosides))

    def __iter__(self):
        yield from self.nucleosides

    def __len__(self):
        return len(self.nucleosides)

    def __repr__(self):
        return f"{{{','.join(self.nucleosides)}}}"


def calculate_diff_errors(mass1, mass2, threshold) -> float:
    if mass1 == mass2:
        return 1.0
    retval = threshold * ((mass1**2 + mass2**2) ** 0.5) / abs(mass1 - mass2)
    # Constrain the maximum relative error to 1!
    # For mass difference very close to zero, the relative error can be very high!
    if retval > 1:
        retval = 1.0
    return retval


def calculate_diff_dp(
    diff, threshold, modification_rate, seq_len, dp_table, breakage_dict
):
    # TODO: Add support for other breakages than 'c/y_c/y'
    explanation_list = [
        entry
        for entry in explain_mass(
            diff,
            dp_table=dp_table,
            seq_len=seq_len,
            max_modifications=round(modification_rate * seq_len),
            threshold=threshold,
            breakage_dict=breakage_dict,
        )
        if entry.breakage == "c/y_c/y"
    ][0].explanations

    # Return None if no explanation was found
    if explanation_list is None:
        return None

    # Return all found explanations
    explanation_list = list(explanation_list)
    return [Explanation(*explanation_list[i]) for i in range(len(explanation_list))]
