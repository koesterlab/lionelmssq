from pathlib import Path
from typing import Tuple

from spectrseqtools.compositions import CompositionList
from spectrseqtools.prediction.composition_inference import (
    CompositionInferrer,
    infer_compositions_with_matrix,
)

ERROR_METHOD = "l1_norm"


def set_output_path(input_path: Path, output_dir: Path) -> Tuple[Path, str]:
    path = input_path.resolve()
    path_dir = path.parent if output_dir is None else output_dir
    path_prefix = path.stem

    return path_dir, path_prefix


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
) -> CompositionList:
    composition_list = infer_compositions_with_matrix(
        diff,
        inferrer=inferrer,
        max_modifications=round(inferrer.seq.modification_rate * inferrer.seq.max_len),
        threshold=threshold,
    ).compositions

    # Return None if no composition was found
    if composition_list is None:
        return CompositionList()

    # Return all found compositions
    return CompositionList.from_list(compositions=composition_list)
