# -*- coding: utf-8 -*-
"""Classification of fragments."""

from itertools import product
from pathlib import Path

import yaml

from spectrseqtools.fragments import RawFragments, StandardUnitFragments
from spectrseqtools.masses import ELEMENT_MASSES, PRECISION

# Set fragmentation dict mode (full vs only c/y)
REDUCED_FRAGMENTATION_DICT = True


# METHOD: Precompute all weight changes caused by fragmentation and adapt the
# target masses accordingly while finding compositions explaining it.
# We consider tags at the 5'- or 3'-end to be possible fragmentation options.
# During classification of a given fragment, for each fragmentation option that
# yields a valid mass (i.e. one that has any valid composition),
# duplicate the fragment and determine its fragmentation-independent standard-unit
# mass by subtracting the weight imposed by the fragmentation.


class FragmentClassifier:
    """Class to classify fragments."""

    def __init__(
        self,
        file_path: Path,
        reduced: bool = REDUCED_FRAGMENTATION_DICT,
        precision: float = PRECISION,
    ):
        """
        Initialize classifier by building dictionary over fragmentation options.

        Parameters
        ----------
        file_path : Path
            Path to meta file.
        reduced : bool
            Flag whether reduced fragmentation list (i.e. only c/y) is used.
        precision : float
            Precision used for (fragmentation) masses.

        """
        element_masses = ELEMENT_MASSES

        with open(file_path, "r", encoding="utf-8") as f:
            meta = yaml.safe_load(f)

        # Read tags from meta file
        start_tag = meta.setdefault("5_prime_tag", 555.1294)
        end_tag = meta.setdefault("3_prime_tag", 455.1491)

        # Initialize dict with masses for 5'-end of fragments
        start_dict = {
            # Remove O from SU and add START tag (without H)
            "START": start_tag - element_masses["O"] - element_masses["H+"],
            # Add H to SU to achieve neutral charge
            "c/y": element_masses["H+"],
        }

        # Initialize dict with masses for 3'-end of fragments
        end_dict = {
            # Remove PO3H from SU and add END tag (without H)
            "END": end_tag
            - element_masses["P"]
            - 3 * element_masses["O"]
            - 2 * element_masses["H+"],
            # Remove H from SU to achieve neutral charge
            "c/y": -element_masses["H+"],
        }

        # Add a/w-, b/x-, and d/z-fragmentation for full dict version
        if not reduced:
            # Add PO3H2 to SU to achieve neutral charge
            start_dict["a/w"] = (
                element_masses["P"] + 3 * element_masses["O"] + 2 * element_masses["H+"]
            )
            # Add P2O to SU to achieve neutral charge
            start_dict["b/x"] = element_masses["P"] + 2 * element_masses["O"]
            # Remove OH from SU to achieve neutral charge
            start_dict["d/z"] = -(element_masses["O"] + element_masses["H+"])

            # Remove PO3H2 from SU to achieve neutral charge
            end_dict["a/w"] = -(
                element_masses["P"] + 3 * element_masses["O"] + 2 * element_masses["H+"]
            )
            # Remove P2O from SU to achieve neutral charge
            end_dict["b/x"] = -(element_masses["P"] + 2 * element_masses["O"])
            # Add OH to SU to achieve neutral charge
            end_dict["d/z"] = element_masses["O"] + element_masses["H+"]

        # Collect all unique fragmentation-related mass combinations in dict
        fragmentation_dict = {}
        for start, end in list(product(start_dict, end_dict)):
            val = int((start_dict[start] + end_dict[end]) / precision)
            if val not in fragmentation_dict:
                fragmentation_dict[val] = []
            fragmentation_dict[val] += [f"{start}_{end}"]

        self.fragmentation_options = fragmentation_dict

    @property
    def start_end_fragmentation(self) -> int:
        """Return integer mass of 'START_END' fragmentation."""
        return [
            mass
            for mass, fragmentation in self.fragmentation_options.items()
            if "START_END" in fragmentation
        ][0]

    def classify(self, fragments: RawFragments) -> StandardUnitFragments:
        """
        Classify raw fragments while standardizing them.

        Parameters
        ----------
        fragments : RawFragments
            Raw fragments.

        Returns
        -------
        StandardUnitFragments
            SU-fragments.

        """
        return fragments.standardize(fragmentation_dict=self.fragmentation_options)
