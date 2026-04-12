# -*- coding: utf-8 -*-
"""Classification of fragments."""

from spectrseqtools.dataclasses import RawFragments, StandardUnitFragments

# METHOD: For each fragmentation option that yields a valid mass (i.e. one that
# has any valid composition) for a given fragment, duplicate the fragment
# and determine its fragmentation-independent standard-unit mass by
# subtracting the weight imposed by the fragmentation.


def classify_fragments(
    fragments: RawFragments,
    fragmentation_dict: dict,
) -> StandardUnitFragments:
    """
    Classify raw fragments while standardizing them.

    Parameters
    ----------
    fragments : RawFragments
        Raw fragments.
    fragmentation_dict : dict
        Dictionary with masses of all considered fragmentation types.

    Returns
    -------
    StandardUnitFragments
        SU-fragments.

    """
    fragments = fragments.standardize(fragmentation_dict=fragmentation_dict)

    # TODO: What is the purpose of the below? It is never used anyway as the
    #  mass would be way too high for a mass spectrometer.
    # # Filter out fragments that have a too high mass
    # fragments = fragments.sort(pl.col("standard_unit_mass")).filter(
    #     pl.col("observed_mass") < mass_cutoff
    # )

    return fragments
