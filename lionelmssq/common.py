from enum import Enum
import re
from typing import Any, Set


#_NUCLEOSIDE_RE = re.compile(r"\d*[ACGUT]")
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
