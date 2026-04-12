import importlib.resources
import os
from pathlib import Path

import pytest
import yaml
from clr_loader import get_mono
from spectrseqtools.cli import predict
from spectrseqtools.dataclasses import Sequence
from spectrseqtools.enums import SolverType
from spectrseqtools.nucleotide_alphabet import NucleotideAlphabet
from spectrseqtools.parsers import PredictionOptions, PreprocessingOptions
from spectrseqtools.plotting import plot_prediction
from spectrseqtools.preprocessing.preprocessing import Preprocessor

rt = get_mono()

_TESTCASES = importlib.resources.files("tests") / "testcases"

TESTS = ["test_01", "test_02", "test_03"]
# TESTS = ["test_01", "test_02", "test_03", "test_04", "test_05", "test_06", "test_07"]


@pytest.mark.parametrize(
    "testcase",
    [tc for tc in _TESTCASES.iterdir() if tc.name in TESTS],
    ids=[tc.name for tc in _TESTCASES.iterdir() if tc.name in TESTS],
)
def test_testcase(testcase):
    # Read additional parameter from meta file
    base_path = Path(_TESTCASES / f"{testcase}")
    with open(base_path / "fragments.meta.yaml", "r") as f:
        meta = yaml.safe_load(f)

    if meta.get("skip"):
        pytest.skip("Testcase is marked as skipped in meta.yaml")

    # Preprocess raw input data if given
    if os.path.isfile(base_path / "fragments.raw"):
        # Preprocess raw data
        Preprocessor(
            options=PreprocessingOptions(
                input=base_path / "fragments.raw",
                meta=base_path / "fragments.meta.yaml",
                output_dir=None,
                alphabet=None,
                charge_range=None,
                min_intensity=None,
                cutoff_percentile=75,
            )
        ).preprocess()
    else:
        # Copy metadata otherwise
        with open(base_path / "fragments.preprocessed.meta.yaml", "w") as f:
            yaml.safe_dump(meta, f)

    prediction = predict(
        PredictionOptions(
            fragments=base_path / "fragments.tsv",
            meta=base_path / "fragments.preprocessed.meta.yaml",
            singletons=base_path / "fragments.singletons.tsv",
            sequence_prediction=base_path / "fragments.prediction.fasta",
            fragment_predictions=base_path / "fragments.prediction.tsv",
            sequence_name=f"{testcase}",
            output_dir=None,
            solver=SolverType.CBC,
            # solver=SolverType.GUROBI,
        )
    )

    # Read true sequence from meta file
    true_seq = Sequence.from_str(meta["true_sequence"])

    print("True sequence =\t\t", true_seq)
    print(
        "Full sequence =\t\t",
        prediction.sequence.to_full_str(
            nucleotide_alphabet=NucleotideAlphabet.from_file()
        ),
    )

    plots = plot_prediction(prediction=prediction, true_seq=true_seq)

    # plots[0].save(base_path / "fragments.plot.start.html")
    # plots[1].save(base_path / "fragments.plot.end.html")
    # plots[2].save(base_path / "fragments.plot.internal.html")
    plots[3].save(base_path / "fragments.plot.html")

    # Save updated meta data
    meta["predicted_sequence"] = prediction.sequence.to_str()
    with open(base_path / "fragments.testing.meta.yaml", "w") as f:
        yaml.safe_dump(meta, f)

    # Assert whether the sequences match
    assert prediction.sequence == true_seq

    # Assert whether observed and predicted mass match for all fragments
    # Note this will only be true for simulated data; experimental data does
    # not have any guarantee accuracy
    # if simulation:
    #     for idx in range(len(prediction.fragments)):
    #         assert abs(
    #             prediction.fragments.item(idx, "standard_unit_mass")
    #             - prediction.fragments.item(idx, "predicted_mass")
    #         ) <= TOLERANCE * prediction.fragments.item(idx, "observed_mass")
