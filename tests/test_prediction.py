import importlib.resources
import os

import pytest

from lionelmssq.cli import select_solver
from lionelmssq.mass_table import DynamicProgrammingTable
from lionelmssq.prediction import Predictor
from lionelmssq.common import parse_nucleosides
from lionelmssq.plotting import plot_prediction
from lionelmssq.fragment_classification import classify_fragments

import polars as pl
import yaml

from lionelmssq.masses import (
    COMPRESSION_RATE,
    TOLERANCE,
    MATCHING_THRESHOLD,
    build_breakage_dict,
    initialize_nucleotide_df,
)

_TESTCASES = importlib.resources.files("tests") / "testcases"

TESTS = ["test_01", "test_02", "test_03"]
# TESTS = ["test_01", "test_02", "test_03", "test_04", "test_05", "test_08"]


@pytest.mark.parametrize(
    "testcase",
    # [tc for tc in _TESTCASES.iterdir() if tc.name not in [".DS_Store"]],
    # ids=[tc.name for tc in _TESTCASES.iterdir() if tc.name not in [".DS_Store"]],
    [tc for tc in _TESTCASES.iterdir() if tc.name in TESTS],
    ids=[tc.name for tc in _TESTCASES.iterdir() if tc.name in TESTS],
)
def test_testcase(testcase):
    base_path = _TESTCASES / testcase
    with open(base_path / "fragments.meta.yaml", "r") as f:
        meta = yaml.safe_load(f)
    if meta.get("skip"):
        pytest.skip("Testcase is marked as skipped in meta.yaml")

    true_seq = parse_nucleosides(meta["true_sequence"])

    input_file = pl.read_csv(base_path / "fragments.tsv", separator="\t")

    if "intensity_cutoff" in meta:
        intensity_cutoff = meta["intensity_cutoff"]
    else:
        intensity_cutoff = 1e4

    if "sequence_mass" in meta:
        seq_mass = meta["sequence_mass"]
    else:
        seq_mass = None

    matching_threshold = MATCHING_THRESHOLD

    # If the left and right columns exist, means that the input file is from a simulation with the sequence of each fragment known!
    if "left" in input_file.columns or "right" in input_file.columns:
        simulation = True

        fragments = pl.read_csv(
            base_path / "fragments.tsv", separator="\t"
        ).with_columns(
            (pl.col("observed_mass").alias("observed_mass")),
            (pl.col("true_mass_with_backbone").alias("true_mass")),
        )

        _, unique_masses, explanation_masses = initialize_nucleotide_df(
            reduce_set=False
        )

        # TODO: Discuss why it doesn't work with the estimated error!
        # matching_threshold, _, _ = estimate_MS_error_matching_threshold(
        #     fragments, unique_masses=unique_masses, simulation=simulation
        # )
        # print(
        #     "Matching threshold (rel error) estimated from singleton masses = ",
        #     matching_threshold,
        # )

        dp_table = DynamicProgrammingTable(
            explanation_masses,
            reduced_table=True,
            reduced_set=False,
            compression_rate=COMPRESSION_RATE,
            tolerance=matching_threshold,
            precision=TOLERANCE,
        )

    else:
        simulation = False

        fragments = pl.read_csv(base_path / "fragments.tsv", separator="\t")

        # TODO: Discuss why it doesn't work with the estimated error!
        # matching_threshold, _, _ = estimate_MS_error_MATCHING_THRESHOLD(
        #     fragment_masses_read, unique_masses=unique_masses, simulation=simulation
        # )
        # print(
        #     "Matching threshold (rel error) estimated from singleton masses = ",
        #     matching_threshold,
        # )

        _, unique_masses, explanation_masses = initialize_nucleotide_df(reduce_set=True)

        dp_table = DynamicProgrammingTable(
            explanation_masses,
            reduced_table=False,
            reduced_set=True,
            compression_rate=COMPRESSION_RATE,
            tolerance=max(matching_threshold, 20e-6),
            # tolerance=matching_threshold,
            precision=TOLERANCE,
        )

    # fragment_masses = pl.Series(fragments.select(pl.col("observed_mass"))).to_list()

    breakage_dict = build_breakage_dict(
        mass_5_prime=meta["label_mass_5T"], mass_3_prime=meta["label_mass_3T"]
    )

    fragments = classify_fragments(
        fragments,
        dp_table=dp_table,
        breakage_dict=breakage_dict,
        output_file_path=base_path / "fragments.standard_unit_fragments.tsv",
        intensity_cutoff=intensity_cutoff,
        seq_mass=seq_mass,
    )

    solver_params = {
        "solver": select_solver(os.environ.get("SOLVER", "cbc")),
        # "solver": select_solver(os.environ.get("SOLVER", "gurobi")),
        "threads": 16,
        "msg": False,
    }

    prediction = Predictor(
        dp_table=dp_table,
        explanation_masses=explanation_masses,
    ).predict(
        fragments=fragments,
        seq_len=len(true_seq),
        solver_params=solver_params,
    )

    print("Predicted sequence =\t", prediction.sequence)
    print("True sequence =\t\t", true_seq)

    if simulation:
        plot_prediction(
            prediction,
            true_seq,
        ).save(base_path / "fragments.plot.html")
        # The above is temporary, until the prediction for the entire intact sequence is fixed!)
    else:
        plot_prediction(prediction, true_seq).save(base_path / "fragments.plot.html")

    meta["predicted_sequence"] = "".join(prediction.sequence)
    with open(base_path / "fragments.meta.yaml", "w") as f:
        yaml.safe_dump(meta, f)

    # Assert whether the sequences match
    assert prediction.sequence == true_seq

    # Assert whether observed and predicted mass match for all fragments
    # Note this will only be true for simulated data; experimental data does
    # not have any guarantee accuracy
    if simulation:
        for idx in range(len(prediction.fragments)):
            assert abs(
                prediction.fragments.item(idx, "standard_unit_mass")
                - prediction.fragments.item(idx, "predicted_mass")
            ) <= matching_threshold * prediction.fragments.item(idx, "observed_mass")
