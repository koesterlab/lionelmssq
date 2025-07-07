from pathlib import Path
from typing import Literal

from lionelmssq.fragment_classification import mark_terminal_fragment_candidates
from lionelmssq.mass_table import DynamicProgrammingTable
from lionelmssq.masses import (
    COMPRESSION_RATE,
    MATCHING_THRESHOLD,
    TOLERANCE,
    initialize_nucleotide_df,
)
from lionelmssq.prediction import Predictor
from tap import Tap
import polars as pl
import yaml


class Settings(Tap):
    fragments: Path  # path to .tsv table with observed fragments to use for prediction
    seq_len: int  # length of the sequence to predict
    fragment_predictions: (
        Path  # path to .tsv table that shall contain the per fragment predictions
    )
    sequence_prediction: (
        Path  # path to .fasta file that shall contain the predicted sequence
    )
    sequence_name: str
    modification_rate: float = 0.5  # maximum percentage of modification in sequence
    solver: Literal["gurobi", "cbc"] = (
        "gurobi"  # solver to use for the optimization problem
    )
    threads: int = 1  # number of threads to use for the optimization problem


def main():
    settings = Settings(underscores_to_dashes=True).parse_args()

    solver_params = {
        "solver": select_solver(settings.solver),
        "threads": settings.threads,
        "msg": False
    }

    fragments = pl.read_csv(settings.fragments, separator="\t")

    simulation = False
    reduce_table = False
    reduce_set = True
    start_tag = 0.0
    end_tag = 0.0
    if "observed_mass" in fragments.columns:
        simulation = True
        reduce_table = True
        reduce_set = False

    _, unique_masses, explanation_masses = initialize_nucleotide_df(
        reduce_set=reduce_set
    )

    dp_table = DynamicProgrammingTable(
        nucleotide_df=explanation_masses,
        compression_rate=int(COMPRESSION_RATE),
        tolerance=MATCHING_THRESHOLD,
        precision=TOLERANCE,
        reduced_table=reduce_table,
        reduced_set=reduce_set,
    )

    if not simulation:
        fragment_dir = settings.fragments.parent
        with open(fragment_dir / "meta.yaml", "r") as f:
            meta = yaml.safe_load(f)

        start_tag = meta["label_mass_5T"]
        end_tag = meta["label_mass_3T"]

        fragments = mark_terminal_fragment_candidates(
            fragments,
            dp_table=dp_table,
            output_file_path=fragment_dir / "fragments_with_classification_marked.tsv",
            # matching_threshold=matching_threshold,
            intensity_cutoff=meta["intensity_cutoff"]
            if "intensity_cutoff" in meta
            else 1e4,
            ms1_mass=meta["sequence_mass"] if "sequence_mass" in meta else None,
        )

    prediction = Predictor(
        fragments=fragments,
        seq_len=settings.seq_len,
        dp_table=dp_table,
        unique_masses=unique_masses,
        explanation_masses=explanation_masses,
        mass_tag_start=start_tag,
        mass_tag_end=end_tag,
    ).predict(
        solver_params=solver_params,
        modification_rate=settings.modification_rate
    )

    # save fragment predictions
    prediction.fragments.write_csv(settings.fragment_predictions, separator="\t")

    # save predicted sequence
    with open(settings.sequence_prediction, "w") as f:
        print(f">{settings.sequence_name}", file=f)
        print("".join(prediction.sequence), file=f)


def select_solver(solver: str):
    match solver:
        case "gurobi":
            return "GUROBI_CMD"
        case "cbc":
            return "PULP_CBC_CMD"
        case _:
            raise NotImplementedError(
                f"Support for '{solver}' is currently not given."
            )