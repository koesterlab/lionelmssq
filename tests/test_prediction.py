import importlib.resources
import os

import pytest
from lionelmssq.prediction import Predictor
from lionelmssq.common import parse_nucleosides
from lionelmssq.plotting import plot_prediction
from lionelmssq.utils import determine_terminal_fragments
import polars as pl
import yaml

from lionelmssq.masses import UNIQUE_MASSES, TOLERANCE, MATCHING_THRESHOLD, ROUND_DECIMAL

_TESTCASES = importlib.resources.files("tests") / "testcases"


# @pytest.mark.parametrize(
#     "testcase", [tc for tc in _TESTCASES.iterdir() if tc.name in ["test_01", "test_02"]]
# )
@pytest.mark.parametrize("testcase", _TESTCASES.iterdir())
def test_testcase(testcase):
    base_path = _TESTCASES / testcase
    with open(base_path / "meta.yaml", "r") as f:
        meta = yaml.safe_load(f)
    if meta.get("skip"):
        pytest.skip("Testcase is marked as skipped in meta.yaml")

    true_seq = parse_nucleosides(meta["true_sequence"])

    input_file = pl.read_csv(base_path / "fragments.tsv", separator="\t")

    # If the left and right columns exist, means that the input file is from a simulation with the sequence of each fragment known!
    if "left" in input_file.columns or "right" in input_file.columns:
        simulation = True

        fragments = pl.read_csv(
            base_path / "fragments.tsv", separator="\t"
        ).with_columns(
            (pl.col("observed_mass_without_backbone").alias("observed_mass")),
            (pl.col("true_nucleoside_mass").alias("true_mass")),
        ).filter(~(pl.col("is_start") & pl.col("is_end")))  #TODO: INCLUDE the cases when both is_start and is_end are True!
        #The above is temporary, until the preeiction for the entire intact sequence is fixed!
        with pl.Config(tbl_rows=30):
            print(fragments)

       # unique_masses = UNIQUE_MASSES
        unique_masses = UNIQUE_MASSES.filter(
            pl.col("nucleoside").is_in(["A", "U", "G", "C"])
        )

        explanation_masses = unique_masses.with_columns(
            (pl.col("monoisotopic_mass") / TOLERANCE)
            .round(0)
            .cast(pl.Int64)
            .alias("tolerated_integer_masses")
        )

    else:
        simulation = False

        label_mass_3T = meta["label_mass_3T"]
        label_mass_5T = meta["label_mass_5T"]

        unique_masses = UNIQUE_MASSES.filter(
            pl.col("nucleoside").is_in(["A", "U", "G", "C"])
        ).with_columns(
            (pl.col("monoisotopic_mass") + 61.95577).alias(
                "monoisotopic_mass"
            )  # Added the appropriate backbone mass!
        )

        explanation_masses = unique_masses.with_columns(
            (pl.col("monoisotopic_mass") / TOLERANCE)
            .round(0)
            .cast(pl.Int64)
            .alias("tolerated_integer_masses")
        )

        fragments = determine_terminal_fragments(
            base_path / "fragments.tsv",
            output_file_path=base_path / "fragments_terminal_marked.tsv",
            label_mass_3T=label_mass_3T,
            label_mass_5T=label_mass_5T,
            explanation_masses=explanation_masses,
            intensity_cutoff=1.2e4, #for test_05
            #intensity_cutoff=5e5, #for test_03
        )
        with pl.Config(tbl_rows=30):
            print(fragments)

    fragment_masses = pl.Series(fragments.select(pl.col("observed_mass"))).to_list()

    prediction = Predictor(
        fragments,
        len(true_seq),
        #os.environ.get("SOLVER", "cbc"),
        os.environ.get("SOLVER", "gurobi"),
        threads=16,
        unique_masses=unique_masses,
        explanation_masses=explanation_masses,
        # "solver": "gurobi" or "cbc"
    ).predict()

    prediction_masses = pl.Series(
        prediction.fragments.select(pl.col("predicted_fragment_mass"))
    ).to_list()

    print("Predicted sequence = ", prediction.sequence)
    print("True sequence = ", true_seq)

    for index,j in enumerate(prediction.fragments.select(pl.col("predicted_fragment_seq")).iter_rows()):
        if all(nuc is None for nuc in j):
            print(index,j,"All None")

    if simulation:
        plot_prediction(prediction, true_seq, fragments.filter(~(pl.col("is_start") & pl.col("is_end")))).save(base_path / "plot.html")
        #TODO: Exclude the cases when both is_start and is_end are True!
        #The above is temporary, until the preeiction for the entire intact sequence is fixed!)
    else:
        plot_prediction(prediction, true_seq).save(base_path / "plot.html")

    meta["predicted_sequence"] = "".join(prediction.sequence)
    with open(base_path / "meta.yaml", "w") as f:
        yaml.safe_dump(meta, f)

    # Assert if the sequences match!
    assert prediction.sequence == true_seq

    # Assert if all the sequence fragments match the predicted fragments in mass at least!
    for i in range(len(fragment_masses)):
        print(f"Fragment {i}: {fragment_masses[i]} vs {prediction_masses[i]}")
        #assert abs(prediction_masses[i] / fragment_masses[i] - 1) <= MATCHING_THRESHOLD

test_testcase("test_06")