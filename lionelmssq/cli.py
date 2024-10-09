from pathlib import Path
from lionelmssq.predict_seq import predict_seq
from tap import Tap
import polars as pl


class Settings(Tap):
    fragments: Path  # path to .tsv table with observed framents to use for prediction
    seq_len: int  # length of the sequence to predict
    fragment_predictions: (
        Path  # path to .tsv table that shall contain the per fragment predictions
    )
    sequence_prediction: Path  # path to .fasta file that shall contain the predicted sequence
    sequence_name: str


def main():
    settings = Settings(underscores_to_dashes=True).parse_args()
    fragments = pl.read_csv(settings.fragments, separator="\t")
    prediction = predict_seq(fragments, settings.seq_len)

    # save fragment predictions
    prediction.fragments.write_csv(settings.fragment_predictions, separator="\t")

    # save predicted sequence
    with open(settings.sequence_prediction, "w") as f:
        print(f">{settings.sequence_name}", file=f)
        print("".join(prediction.sequence))
