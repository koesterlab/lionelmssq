import pytest
import polars as pl

from lionelmssq.mass_explanation import (
    explain_mass,
    explain_mass_with_dp,
    explain_mass_without_breakage,
    explain_mass_recursively_without_breakage,
)
from lionelmssq.masses import (
    EXPLANATION_MASSES,
    START_OPTIONS,
    END_OPTIONS,
    PHOSPHATE_LINK_MASS,
    MASSES,
    TOLERANCE,
)
from lionelmssq.mass_table import DynamicProgrammingTable


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
MOD_RATE = 0.5


@pytest.mark.parametrize("testcase", MASS_SEQ_DICT.items())
@pytest.mark.parametrize("threshold", THRESHOLDS)
def test_testcase(testcase, threshold):
    breakage = list(testcase[1].keys())[0]

    dp_table = DynamicProgrammingTable(
        EXPLANATION_MASSES,
        reduced_table=True,
        reduced_set=False,
        compression_rate=32,
        tolerance=threshold,
        precision=TOLERANCE,
    )

    predicted_mass_explanations = explain_mass(
        testcase[0],
        dp_table=dp_table,
        seq_len=len(testcase[1][breakage]),
        max_modifications=round(MOD_RATE * len(tuple(testcase[1][breakage]))),
    )

    explanations = [
        tuple(solution)
        for expl in predicted_mass_explanations
        if expl.breakage == breakage
        for solution in expl.explanations
    ]

    assert tuple(testcase[1][breakage]) in explanations


@pytest.mark.parametrize("testcase", MASS_SEQ_DICT.items())
@pytest.mark.parametrize("threshold", THRESHOLDS)
def test_testcase_recursively_without_breakage(testcase, threshold):
    breakage = list(testcase[1].keys())[0]

    dp_table = DynamicProgrammingTable(
        EXPLANATION_MASSES,
        reduced_table=True,
        reduced_set=False,
        compression_rate=32,
        tolerance=threshold,
        precision=TOLERANCE,
    )

    predicted_mass_explanations = explain_mass_recursively_without_breakage(
        testcase[0],
        dp_table=dp_table,
        seq_len=len(testcase[1][breakage]),
        max_modifications=round(MOD_RATE * len(tuple(testcase[1][breakage]))),
    ).explanations

    assert predicted_mass_explanations is not None

    explanations = [tuple(expl) for expl in predicted_mass_explanations]

    assert tuple(testcase[1][breakage]) in explanations


WITH_MEMO = [True]
COMPRESSION_RATES = [32]


@pytest.mark.parametrize("testcase", MASS_SEQ_DICT.items())
@pytest.mark.parametrize("compression", COMPRESSION_RATES)
@pytest.mark.parametrize("memo", WITH_MEMO)
@pytest.mark.parametrize("threshold", THRESHOLDS)
def test_testcase_with_dp(testcase, compression, memo, threshold):
    breakage = list(testcase[1].keys())[0]

    dp_table = DynamicProgrammingTable(
        EXPLANATION_MASSES,
        reduced_table=True,
        reduced_set=False,
        compression_rate=compression,
        tolerance=threshold,
        precision=TOLERANCE,
    )

    predicted_mass_explanations = explain_mass_with_dp(
        testcase[0],
        with_memo=memo,
        dp_table=dp_table,
        seq_len=len(testcase[1][breakage]),
        max_modifications=round(MOD_RATE * len(tuple(testcase[1][breakage]))),
        compression_rate=compression,
    )

    explanations = [
        tuple(solution)
        for expl in predicted_mass_explanations
        if expl.breakage == breakage
        for solution in expl.explanations
    ]

    assert tuple(testcase[1][breakage]) in explanations


@pytest.mark.parametrize("testcase", MASS_SEQ_DICT.items())
@pytest.mark.parametrize("compression", COMPRESSION_RATES)
@pytest.mark.parametrize("memo", WITH_MEMO)
@pytest.mark.parametrize("threshold", THRESHOLDS)
def test_testcase_without_breakage(testcase, compression, threshold, memo):
    breakage = list(testcase[1].keys())[0]

    dp_table = DynamicProgrammingTable(
        EXPLANATION_MASSES,
        reduced_table=True,
        reduced_set=False,
        compression_rate=compression,
        tolerance=threshold,
        precision=TOLERANCE,
    )

    predicted_mass_explanations = explain_mass_without_breakage(
        testcase[0],
        dp_table=dp_table,
        seq_len=len(testcase[1][breakage]),
        max_modifications=round(MOD_RATE * len(tuple(testcase[1][breakage]))),
        with_memo=memo,
    ).explanations

    assert predicted_mass_explanations is not None

    explanations = [tuple(expl) for expl in predicted_mass_explanations]

    assert tuple(testcase[1][breakage]) in explanations
