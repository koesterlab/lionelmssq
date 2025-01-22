import importlib.resources
import os

import pytest
from lionelmssq.prediction import Predictor
from lionelmssq.common import parse_nucleosides
from lionelmssq.plotting import plot_prediction
from lionelmssq.utils import determine_terminal_fragments
import polars as pl
import yaml

_TESTCASES = importlib.resources.files("tests") / "testcases"

#MATCHING_THRESHOLD = 10  # Import this from masses.py later!


#@pytest.mark.parametrize(
#    "testcase", [tc for tc in _TESTCASES.iterdir() if tc.name == "test_02"]
#)

def test_testcase(testcase):
    base_path = _TESTCASES / testcase
    with open(base_path / "meta.yaml", "r") as f:
        meta = yaml.safe_load(f)

    true_seq = parse_nucleosides(meta["true_sequence"])
    label_mass_3T = meta["label_mass_3T"]
    label_mass_5T = meta["label_mass_5T"]

    fragments = determine_terminal_fragments(
        base_path / "fragments.tsv",
        output_file_path=base_path / "fragments_terminal_marked.tsv",
        label_mass_3T=label_mass_3T,
        label_mass_5T=label_mass_5T,
        intensity_cutoff=1.2e4,
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

    plot_prediction(prediction, true_seq, fragments).save(base_path / "plot.html")

    print(prediction.sequence, true_seq)

    assert prediction.sequence == true_seq

    # Assert if all the sequence fragments match the predicted fragments in mass at least!
    # for i in range(len(fragment_masses)):
    #    assert abs(fragment_masses[i] - prediction_masses[i]) <= MATCHING_THRESHOLD

    # assert all([abs(fragment_masses[i] - prediction_masses[i]) <= MATCHING_THRESHOLD for i in range(len(fragment_masses))])  #Use is close function here!
    # Check all together!


test_testcase("test_02")