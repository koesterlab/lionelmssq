import importlib.resources
import os
from pathlib import Path

import pytest
import yaml
from clr_loader import get_mono
from spectrseqtools.dataclasses import Sequence
from spectrseqtools.multiplexing import pre_process_multiplexing
from spectrseqtools.parsers import MixturePreprocessingOptions

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
