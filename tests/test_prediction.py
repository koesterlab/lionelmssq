import importlib.resources
import os

import pytest
from lionelmssq.predict_seq import predict_seq
from lionelmssq.common import parse_nucleosides
from lionelmssq.plotting import plot_prediction_with_truth
import polars as pl
import yaml

_TESTCASES = importlib.resources.files("tests") / "testcases"


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

    prediction = predict_seq(
        fragments, len(true_seq), os.environ.get("SOLVER", "cbc"), threads=16
    )

    plot_prediction_with_truth(prediction, true_seq, fragments).save(
        base_path / "plot.html"
    )

    assert prediction.sequence == true_seq
