# import importlib.resources
# import os
import pytest
import polars as pl
# import yaml

from lionelmssq.mass_explanation import explain_mass, explain_mass_with_dp
from lionelmssq.masses import START_OPTIONS, END_OPTIONS, PHOSPHATE_LINK_MASS, MASSES


def get_breakage_weight(breakage: str) -> float:
    start, end = breakage.split("_")[:2]
    return (
        START_OPTIONS.filter(pl.col("name") == start).select("weight").item()
        + END_OPTIONS.filter(pl.col("name") == end).select("weight").item()
    )


def get_seq_weight(seq: tuple) -> float:
    seq_df = pl.DataFrame(data=seq, schema=["name"])
    seq_df = seq_df.with_columns(
        pl.col("name")
        .map_elements(
            lambda x: MASSES.filter(pl.col("nucleoside") == x)
            .get_column("monoisotopic_mass")
            .to_list()[0],
            return_dtype=pl.Float64,
        )
        .alias("mass")
    )

    return round(len(seq) * PHOSPHATE_LINK_MASS + seq_df.select("mass").sum().item(), 5)


TEST_SEQ = [
    {"c/y_c/y": ("A")},
    {"c/y_c/y": ("A", "A")},
    {"c/y_c/y": ("G", "G")},
    {"c/y_c/y": ("C", "C")},
    {"c/y_c/y": ("U", "U")},
    {"c/y_c/y": ("C", "U", "A", "G")},
    # {"c/y_c/y": ("C", "C", "U", "A", "G", "G")},
]

MASS_SEQ_DICT = dict(
    zip(
        [
            get_breakage_weight(list(seq.keys())[0])
            + get_seq_weight(seq[list(seq.keys())[0]])
            for seq in TEST_SEQ
        ],
        TEST_SEQ,
    )
)
THRESHOLDS = [10e-6, 5e-6, 2e-6]


@pytest.mark.parametrize("testcase", MASS_SEQ_DICT.items())
@pytest.mark.parametrize("threshold", THRESHOLDS)
def test_testcase(testcase, threshold):
    predicted_mass_explanations = explain_mass(
        testcase[0], matching_threshold=threshold
    )

    breakage = list(testcase[1].keys())[0]
    explanations = [
        tuple(solution)
        for expl in predicted_mass_explanations
        for solution in expl.explanations
        if expl.breakage == breakage
    ]

    assert tuple(testcase[1][breakage]) in explanations


WITH_MEMO = [True]
COMPRESSION_RATES = [32]


@pytest.mark.parametrize("testcase", MASS_SEQ_DICT.items())
@pytest.mark.parametrize("compression", COMPRESSION_RATES)
@pytest.mark.parametrize("memo", WITH_MEMO)
@pytest.mark.parametrize("threshold", THRESHOLDS)
def test_testcase_with_dp(testcase, compression, memo, threshold):
    predicted_mass_explanations = explain_mass_with_dp(
        testcase[0], with_memo=memo, compression_rate=compression, threshold=threshold
    )

    breakage = list(testcase[1].keys())[0]
    explanations = [
        tuple(solution)
        for expl in predicted_mass_explanations
        for solution in expl.explanations
        if expl.breakage == breakage
    ]

    assert tuple(testcase[1][breakage]) in explanations
