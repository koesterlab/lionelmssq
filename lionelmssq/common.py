from enum import Enum
import re
from typing import Any, Set
from dataclasses import dataclass

MILP_QUASI_ONE_THRESHOLD = 0.9


# _NUCLEOSIDE_RE = re.compile(r"\d*[ACGUT]")
_NUCLEOSIDE_RE = re.compile(r"\d*[ACGU]")


class Explanation:
    def __init__(self, *nucleosides):
        self.nucleosides = tuple(sorted(nucleosides))

    def __iter__(self):
        yield from self.nucleosides

    def __len__(self):
        return len(self.nucleosides)

    def __repr__(self):
        return f"{{{",".join(self.nucleosides)}}}"


@dataclass
class TerminalFragment:
    index: int  # fragment index
    min_end: int  # minimum length of fragment (negative for end fragments)
    max_end: int  # maximum length of fragment (negative for end fragments)


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
