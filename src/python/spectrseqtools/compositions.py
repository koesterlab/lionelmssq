# -*- coding: utf-8 -*-
"""Module for composition-related classes."""

from dataclasses import dataclass, field
from itertools import chain, groupby
from typing import Any, List, Self, Set, Tuple

from spectrseqtools.nucleotide_alphabet import NucleotideAlphabet


class Composition:
    """Class for compositions."""

    def __init__(self, *nucleosides):
        self.nucleosides = tuple(sorted(nucleosides))

    def __iter__(self):
        yield from self.nucleosides

    def __len__(self):
        return len(self.nucleosides)

    def __repr__(self):
        return f"{{{','.join(str(nuc) for nuc in self.nucleosides)}}}"

    def __eq__(self, other):
        return self.nucleosides == other


@dataclass
class CompositionList:
    """Class for lists of compositions corresponding to a given mass."""

    compositions: List[Composition] = field(default_factory=list)

    def __iter__(self):
        return self.compositions.__iter__()

    def __len__(self) -> int:
        return len(self.compositions)

    def __add__(self, other: Any) -> Self:
        if isinstance(other, CompositionList):
            new_compositions = self.compositions + [
                comp for comp in other.compositions if comp not in self.compositions
            ]
            return CompositionList(compositions=new_compositions)
        raise TypeError(
            f"Unsupported operand type(s) for +: {type(self)} and {type(other)}"
        )

    @classmethod
    def from_indices(
        cls, solutions: List[List[int]], alphabet: NucleotideAlphabet
    ) -> Self:
        """
        Initialize composition list from index lists.

        Parameters
        ----------
        solutions : List[List[int]]
            List of nucleotide index lists (representing compositions).
        alphabet : NucleotideAlphabet
            Alphabet of considered nucleotides.

        """
        # Return default if no composition is found
        if len(solutions) == 0:
            return cls()

        # Store the representative tuples for the given indices in a set
        solution_names = set()

        # Convert the masses to their respective representative
        for solution in solutions:
            if len(solution) == 0:
                continue
            solution_names.update([(alphabet.get_rep(entry) for entry in solution)])

        # Return composition set
        return cls.from_list(solution_names)

    @classmethod
    def from_list(cls, compositions: Set[Tuple[str]]) -> Self:
        """Initialize composition list from list of unsorted values."""
        return cls(compositions=[Composition(*comp) for comp in list(compositions)])

    @property
    def nucleotides(self) -> Set[str]:
        """Return set of nucleotides found in any composition in list."""
        return set(chain(*self.compositions))

    def contains_singleton(self) -> bool:
        """Return flag whether any composition is a singleton."""
        if len(self) < 0 and any(len(comp) == 1 for comp in self.compositions):
            return True
        return False

    def group_by_len(self) -> dict:
        """Return dictionary with compositions ordered by their length."""
        return {
            comp_len: set(chain(*comps))
            for comp_len, comps in groupby(self.compositions, len)
        }
