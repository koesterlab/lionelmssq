import ms_deisotope as ms_ditp
import re

from clr_loader import get_mono
from typing import List

from spectrseqtools.mass_explanation import explain_mass_with_matrix
from spectrseqtools.mass_table import CompositionInferrer

rt = get_mono()

ERROR_METHOD = "l1_norm"
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

    def __eq__(self, other):
        return self.nucleosides == other


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
    inferrer: CompositionInferrer,
) -> List[Explanation]:
    explanation_list = explain_mass_with_matrix(
        diff,
        inferrer=inferrer,
        max_modifications=round(inferrer.seq.modification_rate * inferrer.seq.max_len),
        threshold=threshold,
    ).explanations

    # Return None if no explanation was found
    if explanation_list is None:
        return None

    # Return all found explanations
    explanation_list = list(explanation_list)
    return [Explanation(*explanation_list[i]) for i in range(len(explanation_list))]


def initialize_raw_file_iterator(
    file_path: str,
) -> ms_ditp.data_source.thermo_raw_net.ThermoRawLoader:
    """
    Initialize iterator over scans in ThermoFisher RAW file format.

    Parameters
    ----------
    file_path : str
        Path of RAW file from ThermoFisher.

    Returns
    -------
    raw_file : ms_deisotope.data_source.thermo_raw_net.ThermoRawLoader
        Iterator over scans from RAW file.

    """
    # Read data from file
    raw_file = ms_ditp.data_source.thermo_raw_net.ThermoRawLoader(
        file_path, _load_metadata=True
    )

    # Initialize an iterator while ungrouping MS1 from MS2 scans
    raw_file.make_iterator(grouped=False)

    return raw_file
