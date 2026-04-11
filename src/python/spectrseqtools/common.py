import re
from pathlib import Path
from typing import List, Tuple

from spectrseqtools.prediction.composition_inference import (
    infer_compositions_with_matrix,
)
from spectrseqtools.prediction.traceback_matrix import CompositionInferrer

ERROR_METHOD = "l1_norm"
_NUCLEOSIDE_RE = re.compile(r"\d*[ACGU]")


def set_output_path(input_path: Path, output_dir: Path) -> Tuple[Path, str]:
    path = input_path.resolve()
    path_dir = path.parent if output_dir is None else output_dir
    path_prefix = path.stem

    return path_dir, path_prefix


def parse_nucleosides(sequence: str):
    return _NUCLEOSIDE_RE.findall(sequence)


class Composition:
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


def calculate_compositions(
    diff: float,
    threshold: float,
    inferrer: CompositionInferrer,
) -> List[Composition]:
    composition_list = infer_compositions_with_matrix(
        diff,
        inferrer=inferrer,
        max_modifications=round(inferrer.seq.modification_rate * inferrer.seq.max_len),
        threshold=threshold,
    ).compositions

    # Return None if no composition was found
    if composition_list is None:
        return None

    # Return all found compositions
    composition_list = list(composition_list)
    return [Composition(*composition_list[i]) for i in range(len(composition_list))]
