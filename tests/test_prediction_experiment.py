import importlib.resources
import os

import pytest
from lionelmssq.prediction import Predictor
from lionelmssq.common import parse_nucleosides
from lionelmssq.plotting import plot_prediction
from lionelmssq.utils import determine_terminal_fragments
import polars as pl
import yaml

from lionelmssq.masses import UNIQUE_MASSES, TOLERANCE, MATCHING_THRESHOLD

unique_masses = UNIQUE_MASSES.filter(
    pl.col("nucleoside").is_in(["A", "U", "G", "C"])
).with_columns(
    (pl.col("monoisotopic_mass") + 61.95577).alias(
        "monoisotopic_mass"
    )  # Added the appropriate backbone mass!
)

# unique_masses = UNIQUE_MASSES.with_columns(
#    (pl.col("monoisotopic_mass") + 61.95577).alias("monoisotopic_mass") #Added the appropriate backbone mass!
# )

explanation_masses = unique_masses.with_columns(
    (pl.col("monoisotopic_mass") / TOLERANCE)
    .round(0)
    .cast(pl.Int64)
    .alias("tolerated_integer_masses")
)

_TESTCASES = importlib.resources.files("tests") / "testcases"


@pytest.mark.parametrize(
    "testcase", [tc for tc in _TESTCASES.iterdir() if tc.name in ["test_03"]]
)
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
        explanation_masses=explanation_masses,
        # intensity_cutoff=1.2e4,
        intensity_cutoff=5e5,
    )
    with pl.Config(tbl_rows=30):
        print(fragments)

    fragment_masses = pl.Series(fragments.select(pl.col("observed_mass"))).to_list()

    prediction = Predictor(
        fragments,
        len(true_seq),
        os.environ.get("SOLVER", "cbc"),
        threads=16,
        unique_masses=unique_masses,
        # fragments, len(true_seq), os.environ.get("SOLVER", "gurobi"), threads=16
    ).predict()

    prediction_masses = pl.Series(
        prediction.fragments.select(pl.col("observed_mass"))
    ).to_list()

    print("Predicted sequence = ", prediction.sequence)
    print("True sequence = ", true_seq)

    plot_prediction(prediction, true_seq).save(base_path / "plot.html")

    assert prediction.sequence == true_seq

    # Assert if all the sequence fragments match the predicted fragments in mass at least!
    for i in range(len(fragment_masses)):
        assert abs(prediction_masses[i] / fragment_masses[i] - 1) <= MATCHING_THRESHOLD

    meta["predicted_sequence"] = "".join(prediction.sequence)
    with open(base_path / "meta.yaml", "w") as f:
        yaml.safe_dump(meta, f)
