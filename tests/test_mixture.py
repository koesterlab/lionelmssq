import importlib.resources
import os
from pathlib import Path

import pytest
import yaml
from clr_loader import get_mono
from spectrseqtools.dataclasses import Sequence
from spectrseqtools.enums import SolverType
from spectrseqtools.error_calculator import ErrorCalculator
from spectrseqtools.multiplexing import pre_process_multiplexing, predict_multiplexing
from spectrseqtools.nucleotide_alphabet import NucleotideAlphabet
from spectrseqtools.parsers import MixturePreprocessingOptions, PredictionOptions

rt = get_mono()

_TESTCASES = importlib.resources.files("tests") / "mixture_testcases"


@pytest.mark.parametrize(
    "testcase",
    [tc for tc in _TESTCASES.iterdir()],
    ids=[tc.name for tc in _TESTCASES.iterdir()],
)
def test_preprocess_mixture(testcase):
    # Read additional parameter from meta file
    base_path = Path(_TESTCASES / f"{testcase}")
    with open(base_path / "fragments.meta.yaml", "r") as f:
        meta = yaml.safe_load(f)
        if "true_sequence" not in meta:
            meta["true_sequence"] = "".join(meta["true_sequences"])

    if meta.get("skip_preprocessing"):
        pytest.skip("Testcase is marked as skipped in meta.yaml")

    # Preprocess raw input data if given
    if os.path.isfile(base_path / "fragments.raw"):
        # Preprocess raw data
        _, singletons, _ = pre_process_multiplexing(
            options=MixturePreprocessingOptions(
                input=base_path / "fragments.raw",
                meta=base_path / "fragments.meta.yaml",
                output_dir=base_path,
                min_time=10.0,
                max_time=15.0,
                window_size=0.2,
            )
        )
        # Read true sequence from meta file
        true_seq = Sequence.from_str(meta["true_sequence"]).sequence

        # Assert whether the sequences match
        print(singletons)
        singleton_set = set(singletons.get_column("id").to_list())
        for nuc in set(true_seq):
            assert nuc in singleton_set
    else:
        # Copy metadata otherwise
        with open(base_path / "fragments.preprocessed.meta.yaml", "w") as f:
            yaml.safe_dump(meta, f)


@pytest.mark.parametrize(
    "testcase",
    [tc for tc in _TESTCASES.iterdir()],
    ids=[tc.name for tc in _TESTCASES.iterdir()],
)
def test_predict_mixture(testcase):
    # Read additional parameter from meta file
    base_path = Path(_TESTCASES / f"{testcase}")
    with open(base_path / "fragments.meta.yaml", "r") as f:
        meta = yaml.safe_load(f)
        if "true_sequence" not in meta:
            meta["true_sequence"] = "".join(meta["true_sequences"])

    if meta.get("skip_prediction"):
        pytest.skip("Testcase is marked as skipped in meta.yaml")

    alphabet_path = base_path / "fragments.singletons.tsv"
    prediction = predict_multiplexing(
        PredictionOptions(
            fragments=base_path / "fragments.tsv",
            meta=base_path / "fragments.preprocessed.meta.yaml",
            alphabet=alphabet_path,
            sequence_prediction=base_path / "fragments.prediction.fasta",
            fragment_predictions=base_path / "fragments.prediction.tsv",
            sequence_name=f"{meta['identity']}",
            output_dir=base_path,
            intensity_cutoff_percentile=70,
            # solver=SolverType.CBC,
            solver=SolverType.GUROBI,
            # solver=SolverType.HIGHS,
        )
    )

    if "true_sequences" in meta:
        # Read true sequence from meta file
        for true_seq in meta["true_sequences"]:
            true_seq = Sequence.from_str(true_seq)
            print("True sequence =\t\t", true_seq)

            found_match = False
            for pred_seq in prediction:
                pred_seq = Sequence.from_str(pred_seq)

                if true_seq == pred_seq:
                    print(
                        "Full sequence =\t\t",
                        pred_seq.fmt(
                            nucleotide_alphabet=NucleotideAlphabet.from_file(
                                error=ErrorCalculator.with_metric()
                            )
                        ),
                    )
                    found_match = True
                    break

            if not found_match:
                assert 2 == 1

    # Assert whether the sequences match
    assert prediction.sequence == true_seq
