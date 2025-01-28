import importlib.resources
import os

import pytest
from lionelmssq.prediction import Predictor
from lionelmssq.common import parse_nucleosides
from lionelmssq.plotting import plot_prediction
import polars as pl
import yaml

from lionelmssq.masses import MATCHING_THRESHOLD

_TESTCASES = importlib.resources.files("tests") / "testcases"


@pytest.mark.parametrize(
    "testcase", [tc for tc in _TESTCASES.iterdir() if tc.name in ["test_01", "test_02"]]
)
def test_testcase(testcase):
    base_path = _TESTCASES / testcase
    with open(base_path / "meta.yaml", "r") as f:
        meta = yaml.safe_load(f)

    true_seq = parse_nucleosides(meta["true_sequence"])

    print(true_seq)

    fragments = pl.read_csv(base_path / "fragments.tsv", separator="\t").with_columns(
        # (pl.col("left") == 0).alias("is_start"),
        # ((pl.col("right")) == len(true_seq)).alias("is_end"),
        (pl.col("observed_mass_without_backbone").alias("observed_mass")),
        (pl.col("true_nucleoside_mass").alias("true_mass")),
    )
    with pl.Config(tbl_rows=30):
        print(fragments)

    fragment_masses = pl.Series(fragments.select(pl.col("observed_mass"))).to_list()

    prediction = Predictor(
        fragments, len(true_seq), os.environ.get("SOLVER", "cbc"), threads=16
    ).predict()

    prediction_masses = pl.Series(
        prediction.fragments.select(pl.col("observed_mass"))
    ).to_list()

    print("Predicted sequence = ", prediction.sequence)
    print("True sequence = ", true_seq)

    plot_prediction(prediction, true_seq, fragments).save(
        base_path / "plot.html"
    )

    assert prediction.sequence == true_seq

    # Assert if all the sequence fragments match the predicted fragments in mass at least!
    for i in range(len(fragment_masses)):
        assert abs(prediction_masses[i] / fragment_masses[i] - 1) <= MATCHING_THRESHOLD