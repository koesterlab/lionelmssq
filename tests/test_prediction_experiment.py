import importlib.resources
import os

import pytest
from lionelmssq.prediction import Predictor
from lionelmssq.common import parse_nucleosides
from lionelmssq.plotting import plot_prediction_with_truth
import polars as pl
import yaml

_TESTCASES = importlib.resources.files("tests") / "testcases"

MATCHING_THRESHOLD = 10  # Import this from masses.py later!


@pytest.mark.parametrize("testcase", _TESTCASES.iterdir())
def test_testcase(testcase):
    base_path = _TESTCASES / testcase
    with open(base_path / "meta.yaml", "r") as f:
        meta = yaml.safe_load(f)

    true_seq = parse_nucleosides(meta["true_sequence"])

    fragments = pl.read_csv(base_path / "fragments.tsv", separator="\t").with_columns(
        (pl.col("left") == 0).alias("is_start"),
        ((pl.col("right")) == len(true_seq)).alias("is_end"),
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

    plot_prediction_with_truth(prediction, true_seq, fragments).save(
        base_path / "plot.html"
    )

    print(prediction.sequence, true_seq)

    assert prediction.sequence == true_seq

    # Assert if all the sequence fragments match the predicted fragments in mass at least!
    for i in range(len(fragment_masses)):
        assert abs(fragment_masses[i] - prediction_masses[i]) <= MATCHING_THRESHOLD

    # assert all([abs(fragment_masses[i] - prediction_masses[i]) <= MATCHING_THRESHOLD for i in range(len(fragment_masses))])  #Use is close function here!
    # Check all together!
