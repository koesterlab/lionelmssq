from itertools import combinations
from lionelmssq.masses import UNIQUE_MASSES
import polars as pl

# TODO:
# also consider that the observations are not complete and that we probably don't see all the letters as diffs or singletons.
# Hence, maybe do the following: solver first with the reduced alphabet, and if the optimization does not yield a sufficiently
# good result, then try again with an extended alphabet.



def reduce_alphabet_by_fragments(fragments: pl.DataFrame):
    start_masses = fragments.filter(pl.col("is_start")).get_column("observed_mass")
    end_masses = fragments.filter(pl.col("is_end")).get_column("observed_mass")

    # determine mass diffs
    diffs = sorted(
        _mass_diffs(start_masses)
        | _mass_diffs(end_masses)
        | _singular_masses(fragments.get_column("observed_mass"))
    )

    # explain mass diffs
    explanations = _explain_diffs(diffs)

    # unexplained differences
    # unexplained = [
    #     diff
    #     for diff, expls in explanations.items()
    #     if not expls and diff > _MIN_PLAUSIBLE_NUCLEOSIDE_DIFF
    # ]
    # TODO can we do something with the unexplained differences?
    # for example consider more than two bases as difference?
    # at least we should log them and warn the user

    def get_nucleosides(explanations):
        for expls in explanations.values():
            for expl in expls:
                if isinstance(expl, tuple):
                    yield from expl
                else:
                    yield expl

    observed_nucleosides = set(get_nucleosides(explanations))

    reduced = UNIQUE_MASSES.filter(pl.col("nucleoside").is_in(observed_nucleosides))

    return reduced


# TODO: the -2.0 should be informed by the variance in the measurements
_MIN_PLAUSIBLE_NUCLEOSIDE_DIFF = (
    UNIQUE_MASSES.select(pl.col("monoisotopic_mass").min()).item() - 2.0
)
_MAX_PLAUSILE_NUCLEOSIDE_DIFF = (
    UNIQUE_MASSES.select(pl.col("monoisotopic_mass").max()).item() + 2.0
)


def _mass_diffs(masses):
    """Return observed mass differences."""
    diffs = set(masses[1:] - masses[:-1])
    return diffs


def _singular_masses(masses):
    """Return observed masses that are plausible to be a single nucleoside."""
    return set(mass for mass in masses if mass <= _MAX_PLAUSILE_NUCLEOSIDE_DIFF)


def _is_similar(mass_a, mass_b):
    """Return whether two masses are similar enough to be considered the same nucleoside."""
    return abs(mass_a - mass_b) < 1.0


def _explain_diffs(diffs):
    """Return explanations for given mass differences."""

    # explain with single nucleosides
    explanations = {
        diff: [
            item["nucleoside"]
            for item in UNIQUE_MASSES.iter_rows(named=True)
            if _is_similar(diff, item["monoisotopic_mass"])
        ]
        for diff in diffs
    }
    # explain with two nucleosides
    explanations.update(
        {
            diff: [
                (item_a["nucleoside"], item_b["nucleoside"])
                for item_a, item_b in combinations(
                    UNIQUE_MASSES.iter_rows(named=True), 2
                )
                if _is_similar(
                    diff, item_a["monoisotopic_mass"] + item_b["monoisotopic_mass"]
                )
            ]
            for diff in diffs
            if not explanations[diff]
        }
    )
    return explanations
