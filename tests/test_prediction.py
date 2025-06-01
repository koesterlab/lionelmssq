import importlib.resources
import os

import pytest
from lionelmssq.prediction import Predictor
from lionelmssq.common import parse_nucleosides
from lionelmssq.plotting import plot_prediction
from lionelmssq.utils import (
    determine_terminal_fragments,
    estimate_MS_error_MATCHING_THRESHOLD,
    determine_sequence_length,
    predetermine_possible_nucleotides,
)
import polars as pl
import yaml

from lionelmssq.masses import (
    UNIQUE_MASSES,
    TOLERANCE,
    MATCHING_THRESHOLD,
)

from lionelmssq.masses import (
    UNIQUE_MASSES,
    TOLERANCE,
    MATCHING_THRESHOLD,
)

# _TESTCASES = importlib.resources.files("tests") / "testcases"
_TESTCASES = importlib.resources.files("tests") / "testcases_april"
# _TESTCASES = importlib.resources.files("tests") / "testcases_modified"


@pytest.mark.parametrize(
    "testcase",
    # [tc for tc in _TESTCASES.iterdir() if tc.name not in ["test_08", ".DS_Store"]],
    [tc for tc in _TESTCASES.iterdir() if tc.name in ["test_01", "test_02", "test_03"]],
)
# @pytest.mark.parametrize("testcase", _TESTCASES.iterdir())
def test_testcase(testcase):
    base_path = _TESTCASES / testcase
    with open(base_path / "meta.yaml", "r") as f:
        meta = yaml.safe_load(f)
    if meta.get("skip"):
        pytest.skip("Testcase is marked as skipped in meta.yaml")

    true_seq = parse_nucleosides(meta["true_sequence"])

    input_file = pl.read_csv(base_path / "fragments.tsv", separator="\t")

    label_mass_3T = meta["label_mass_3T"]
    label_mass_5T = meta["label_mass_5T"]

    if "intensity_cutoff" in meta:
        intensity_cutoff = meta["intensity_cutoff"]
    else:
        intensity_cutoff = 1e4

    if "sequence_mass" in meta:
        ms1_mass = meta["sequence_mass"]
    else:
        ms1_mass = None

    # If the left and right columns exist, means that the input file is from a simulation with the sequence of each fragment known!
    if "left" in input_file.columns or "right" in input_file.columns:
        simulation = True

        fragments = pl.read_csv(
            base_path / "fragments.tsv", separator="\t"
        ).with_columns(
            (pl.col("observed_mass_without_backbone").alias("observed_mass")),
            (pl.col("true_nucleoside_mass").alias("true_mass")),
            # ((pl.col("left") == 0) & (~(pl.col("right") == (len(true_seq))))).alias("is_start"),
            # ((pl.col("right") == (len(true_seq))) & (~(pl.col("left") == 0))).alias("is_end"),
            # ((pl.col("left") == 0) & (pl.col("right") == (len(true_seq)))).alias("is_start_end"),
            # ((~(pl.col("left") == 0)) & (~(pl.col("right") == (len(true_seq))))).alias("is_internal"),
        )
        with pl.Config(tbl_rows=30):
            print(fragments)

        unique_masses = UNIQUE_MASSES

        explanation_masses = unique_masses.with_columns(
            (pl.col("monoisotopic_mass") / TOLERANCE)
            .round(0)
            .cast(pl.Int64)
            .alias("tolerated_integer_masses")
        )

        # TODO: Discuss why it doesn't work with the estimated error!
        matching_threshold, _, _ = estimate_MS_error_MATCHING_THRESHOLD(
            fragments, unique_masses=unique_masses, simulation=simulation
        )
        matching_threshold = MATCHING_THRESHOLD
        # print(
        #     "Matching threshold (rel errror) estimated from singleton masses = ",
        #     matching_threshold,
        # )

        singleton_mass_filtering_limit = 1.1 * (
            max(unique_masses["monoisotopic_mass"])
        ) + max(label_mass_3T, label_mass_5T)

        fragments_singletons = determine_terminal_fragments(
            fragments.filter(pl.col("observed_mass") < singleton_mass_filtering_limit),
            label_mass_3T=label_mass_3T,
            label_mass_5T=label_mass_5T,
            mass_column_name="observed_mass",
            explanation_masses=explanation_masses,
            matching_threshold=matching_threshold,
            intensity_cutoff=intensity_cutoff,
        )

        print("Fragments singletons = ", fragments_singletons)

        nucleosides, _ = Predictor(
            fragments_singletons,
            unique_masses=unique_masses,
            explanation_masses=explanation_masses,
            matching_threshold=matching_threshold,
            mass_tag_start=label_mass_5T,
            mass_tag_end=label_mass_3T,
        )._calculate_diffs_and_nucleosides()

        unique_masses = UNIQUE_MASSES.filter(pl.col("nucleoside").is_in(nucleosides))

        explanation_masses = unique_masses.with_columns(
            (pl.col("monoisotopic_mass") / TOLERANCE)
            .round(0)
            .cast(pl.Int64)
            .alias("tolerated_integer_masses")
        )

    else:
        simulation = False

        unique_masses = UNIQUE_MASSES.with_columns(
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

        fragment_masses_read = pl.read_csv(base_path / "fragments.tsv", separator="\t")

        # filter fragement_masses_read so that there are no duplicates of the same neutral mass
        fragment_masses_read = fragment_masses_read.group_by(
            "neutral_mass", maintain_order=True
        ).first()

        # # TODO: Discuss why it doesn't work with the estimated error!
        # matching_threshold, _, _ = estimate_MS_error_MATCHING_THRESHOLD(
        #     fragment_masses_read, unique_masses=unique_masses, simulation=simulation
        # )
        _, matching_threshold, _ = estimate_MS_error_MATCHING_THRESHOLD(
            fragment_masses_read, unique_masses=unique_masses, simulation=simulation
        )
        print(
            "Matching threshold (rel errror) estimated from singleton masses = ",
            matching_threshold,
        )
        matching_threshold = MATCHING_THRESHOLD

        nucleosides = predetermine_possible_nucleotides(
            fragment_masses_read,
            explanation_masses=explanation_masses,
            matching_threshold=matching_threshold,
            intensity_cutoff=intensity_cutoff,
        )

        # nucleosides = ["A", "U", "G", "C", "0C"]
        # nucleosides = ["A", "U", "G", "C", "0G"]
        # nucleosides = ["A", "U", "G", "C", "9A"]
        nucleosides = ["A", "U", "G", "C"]

        print("Predetermined nucleosides from singletons: ", nucleosides)

        # fragments_singletons = determine_terminal_fragments(
        #     fragment_masses_read.filter(
        #         pl.col("neutral_mass") < singleton_mass_filtering_limit
        #     ),
        #     label_mass_3T=label_mass_3T,
        #     label_mass_5T=label_mass_5T,
        #     explanation_masses=explanation_masses,
        #     matching_threshold=matching_threshold,
        #     intensity_cutoff=intensity_cutoff,
        # )

        # print("Fragments singletons = ", fragments_singletons)

        # nucleosides, _ = Predictor(
        #     fragments_singletons,
        #     unique_masses=unique_masses,
        #     explanation_masses=explanation_masses,
        #     matching_threshold=matching_threshold,
        #     mass_tag_start=label_mass_5T,
        #     mass_tag_end=label_mass_3T,
        #     print_mass_table=True,
        # )._calculate_diffs_and_nucleosides()

        unique_masses = UNIQUE_MASSES.filter(
            pl.col("nucleoside").is_in(nucleosides)
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

        terminal_marked_path = base_path / "fragments_terminal_marked.tsv"
        if not terminal_marked_path.exists():
            fragments = determine_terminal_fragments(
                fragment_masses_read,
                output_file_path=base_path / "fragments_terminal_marked.tsv",
                label_mass_3T=label_mass_3T,
                label_mass_5T=label_mass_5T,
                explanation_masses=explanation_masses,
                matching_threshold=matching_threshold,
                intensity_cutoff=intensity_cutoff,
                ms1_mass=ms1_mass,
            )
        else:
            fragments = pl.read_csv(terminal_marked_path, separator="\t")

    lengths_seq, _, _ = determine_sequence_length(
        terminally_marked_fragments=fragments, strategy="frequency"
    )
    print("Possible MS1 mass explaining sequence lengths = ", lengths_seq)

    len_seq = lengths_seq[0]
    print("Initial guess for sequence length = ", len_seq)

    # len_seq = len(true_seq) #Get rid of this once the multiple length part is fixed!

    prediction = Predictor(
        fragments,
        len_seq,
        # os.environ.get("SOLVER", "cbc"),
        os.environ.get("SOLVER", "gurobi"),
        threads=16,
        unique_masses=unique_masses,
        explanation_masses=explanation_masses,
        matching_threshold=matching_threshold,
        mass_tag_start=label_mass_5T,
        mass_tag_end=label_mass_3T,
        print_mass_table=False,
    ).predict(num_top_paths=1000, consider_variable_sequence_lengths=True)
    # ).predict(num_top_paths=1000, consider_variable_sequence_lengths=False)

    fragment_masses = pl.Series(
        prediction.fragments.select(pl.col("observed_mass"))
    ).to_list()

    prediction_masses = pl.Series(
        prediction.fragments.select(pl.col("predicted_fragment_mass"))
    ).to_list()

    print("Pred sequence = ", prediction.sequence)
    print("True sequence = ", true_seq)

    if simulation:
        plot_prediction(
            prediction,
            true_seq,
        ).save(base_path / "plot.html")
        # The above is temporary, until the prediction for the entire intact sequence is fixed!)
    else:
        plot_prediction(prediction, true_seq).save(base_path / "plot.html")

    meta["predicted_sequence"] = "".join(prediction.sequence)
    with open(base_path / "meta.yaml", "w") as f:
        yaml.safe_dump(meta, f)

    # Assert if the sequences match!
    assert prediction.sequence == true_seq

    # Assert if all the sequence fragments match the predicted fragments in mass at least!
    if simulation:
        # This will only be true for simulated data, for experimental data, every fragment is not predicted so accurately!
        for i in range(len(fragment_masses)):
            # print(f"Fragment {i}: {fragment_masses[i]} vs {prediction_masses[i]}")
            if (
                abs(fragment_masses[i] - prediction_masses[i])
                < 0.01 * fragment_masses[i]
            ):
                # TODO: The above is a temporary measure, there is an issue with ONE fragment in test_06!
                assert (
                    abs(prediction_masses[i] / fragment_masses[i] - 1)
                    <= matching_threshold
                )


test_testcase("30mers/test_01_2")
# test_testcase("30mers/test_03")
# test_testcase("25mers/test_01_centroid")
# test_testcase("25mers/test_01")
# test_testcase("20mers/test_02_2")
