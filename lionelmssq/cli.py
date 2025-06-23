from pathlib import Path
from typing import Literal
from lionelmssq.prediction import Predictor
from tap import Tap
import polars as pl


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
    fragments = pl.read_csv(settings.fragments, separator="\t")
    prediction = Predictor(
        fragments, settings.seq_len, settings.solver, settings.threads
    ).predict(settings.modification_rate)

    # save fragment predictions
    prediction.fragments.write_csv(settings.fragment_predictions, separator="\t")

    # save predicted sequence
    with open(settings.sequence_prediction, "w") as f:
        print(f">{settings.sequence_name}", file=f)
        print("".join(prediction.sequence), file=f)
