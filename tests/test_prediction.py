import importlib.resources
import os
import polars as pl
import yaml
import pytest
from clr_loader import get_mono

from spectrseqtools.cli import format_sequence_to_full_version, select_solver
from spectrseqtools.mass_table import CompositionInferrer, SequenceInformation
from spectrseqtools.prediction import Predictor
from spectrseqtools.common import parse_nucleosides
from spectrseqtools.plotting import plot_prediction
from spectrseqtools.fragment_classification import classify_fragments
from spectrseqtools.preprocessing import preprocess
from spectrseqtools.masses import (
    COMPRESSION_RATE,
    DEFAULT_INTENSITY_CUTOFF,
    NUC_REPS,
    NUCLEOTIDE_DF,
    PRECISION,
    TOLERANCE,
    UNMODIFIED_BASES,
    build_fragmentation_dict,
)

rt = get_mono()

_TESTCASES = importlib.resources.files("tests") / "testcases"

TESTS = ["test_01", "test_02", "test_03"]
# TESTS = ["test_01", "test_02", "test_03", "test_04", "test_05", "test_06", "test_07"]


@pytest.mark.parametrize(
    "testcase",
    # [tc for tc in _TESTCASES.iterdir() if tc.name not in [".DS_Store"]],
    # ids=[tc.name for tc in _TESTCASES.iterdir() if tc.name not in [".DS_Store"]],
    [tc for tc in _TESTCASES.iterdir() if tc.name in TESTS],
    ids=[tc.name for tc in _TESTCASES.iterdir() if tc.name in TESTS],
)
def test_testcase(testcase):
    # Read additional parameter from meta file
    base_path = _TESTCASES / testcase
    with open(base_path / "fragments.meta.yaml", "r") as f:
        meta = yaml.safe_load(f)

    if meta.get("skip"):
        pytest.skip("Testcase is marked as skipped in meta.yaml")

    # Set parameters for LP solver
    solver_params = {
        "fixed": {
            "solver": select_solver(os.environ.get("SOLVER", "cbc")),
            # "solver": select_solver(os.environ.get("SOLVER", "gurobi")),
            "threads": 1,
            "msg": False,
        },
        "timeLimit(short)": 5,
        "timeLimit(long)": 60,
    }

    # Differentiate between raw and already preprocessed input data
    if os.path.isfile(base_path / "fragments.raw"):
        # Preprocess raw data
        fragments, singletons, meta = preprocess(
            file_path=base_path / "fragments.raw",
            deconvolution_params={},
            meta_params=meta,
        )

        # Save preprocessed fragments
        fragments.write_csv(base_path / "fragments.tsv", separator="\t")

        # Save singletons detected from raw data
        singletons.write_csv(base_path / "fragments.singletons.tsv", separator="\t")
    else:
        # Read already preprocessed fragments
        fragments = pl.read_csv(
            base_path / "fragments.tsv", separator="\t"
        ).with_columns(
            (pl.col("observed_mass").alias("observed_mass")),
            (pl.col("true_mass_with_backbone").alias("true_mass")),
        )

        # Read singletons if given
        singletons = None
        if os.path.isfile(base_path / "fragments.singletons.tsv"):
            singletons = pl.read_csv(
                base_path / "fragments.singletons.tsv", separator="\t"
            )

    print("Singletons identified during preprocessing:", singletons)
    print()

    intensity_cutoff = (
        meta["intensity_cutoff"]
        if ("intensity_cutoff" in meta)
        else DEFAULT_INTENSITY_CUTOFF
    )
    nucleotide_df = NUCLEOTIDE_DF
    tolerance = TOLERANCE

    # Filter by singletons
    if singletons is not None:
        # Map singletons to their mass representative
        singletons = singletons.with_columns(
            pl.col("id").replace_strict(NUC_REPS).alias("id")
        )

        # Select only bases found in singletons
        nucleotide_df = nucleotide_df.with_columns(
            pl.when(
                pl.col("representative").is_in(
                    singletons.get_column("id").to_list()
                )
            )
            .then(pl.col("modification_rate"))
            .otherwise(pl.lit(0.0))
            .alias("modification_rate")
        )

    # Ensure modification rates of unmodified bases are set to 1
    nucleotide_df = nucleotide_df.with_columns(
        pl.when(~pl.col("representative").is_in(UNMODIFIED_BASES))
        .then(pl.col("modification_rate"))
        .otherwise(pl.lit(1.0))
        .alias("modification_rate")
    )

    # TODO: Discuss why it doesn't work with the estimated error!
    # tolerance, _, _ = estimate_MS_error_tolerance(
    #     fragments, unique_masses=unique_masses, simulation=simulation
    # )
    # print(
    #     "Tolerance (rel error) estimated from singleton masses = ",
    #     tolerance,
    # )

    # Build fragmentation dict
    fragmentation_dict = build_fragmentation_dict(
        mass_5_prime=meta["label_mass_5T"], mass_3_prime=meta["label_mass_3T"]
    )

    # Standardize sequence mass (remove START_END fragmentation to gain SU mass)
    seq_mass_obs = meta["sequence_mass"]
    seq_mass_su = (
        seq_mass_obs
        - [
            mass * PRECISION
            for mass in fragmentation_dict
            if "START_END" in fragmentation_dict[mass]
        ][0]
    )

    # Initialize SequenceInformation class
    seq_info = SequenceInformation(
        max_len=int(
            seq_mass_su
            / PRECISION
            / min(
                pl.Series(
                    nucleotide_df.filter(pl.col("modification_rate") > 0.0).select(
                        "integer_mass"
                    )
                ).to_list()
            )
        ),
        su_mass=seq_mass_su,
        obs_mass=seq_mass_obs,
        modification_rate=0.5,
    )

    # Initialize CompositionInferrer class
    inferrer = CompositionInferrer(
        nucleotide_df=nucleotide_df,
        compression_rate=COMPRESSION_RATE,
        tolerance=tolerance,
        precision=PRECISION,
        seq=seq_info,
    )

    print("Alphabet after singleton reduction:")
    inferrer.print_alphabet()
    print()

    # Classify preprocessed fragments
    fragments = classify_fragments(
        fragment_masses=fragments,
        inferrer=inferrer,
        fragmentation_dict=fragmentation_dict,
        output_file_path=base_path / "fragments.standard_unit_fragments.tsv",
        intensity_cutoff=intensity_cutoff,
    )

    # Predict sequence
    prediction = Predictor(
        inferrer=inferrer,
        nucleotide_df=nucleotide_df,
    ).predict(
        fragments=fragments,
        solver_params=solver_params,
    )

    # Read true sequence from meta file
    true_seq = parse_nucleosides(meta["true_sequence"])

    print("Predicted sequence =\t", prediction.sequence)
    print("True sequence =\t\t", true_seq)

    print(
        "Full sequence =\t\t", format_sequence_to_full_version(seq=prediction.sequence)
    )

    plots = plot_prediction(prediction=prediction, true_seq=true_seq)

    # plots[0].save(base_path / "fragments.plot.start.html")
    # plots[1].save(base_path / "fragments.plot.end.html")
    # plots[2].save(base_path / "fragments.plot.internal.html")
    plots[3].save(base_path / "fragments.plot.html")

    # Save updated meta data
    meta["predicted_sequence"] = "".join(prediction.sequence)
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
    #         ) <= tolerance * prediction.fragments.item(idx, "observed_mass")
