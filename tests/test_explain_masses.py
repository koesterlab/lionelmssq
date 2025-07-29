import pytest
import polars as pl

from lionelmssq.mass_explanation import (
    explain_mass_with_recursion,
    explain_mass_with_table,
)
from lionelmssq.masses import (
    EXPLANATION_MASSES,
    PHOSPHATE_LINK_MASS,
    MASSES,
    TOLERANCE,
)
from lionelmssq.mass_table import DynamicProgrammingTable


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
    tuple("A"),
    ("A", "A"),
    ("G", "G"),
    ("C", "C"),
    ("U", "U"),
    ("C", "U", "A", "G"),
    ("C", "C", "U", "A", "G", "G"),
]

MASS_SEQ_DICT = dict(
    zip(
        [get_seq_weight(seq) for seq in TEST_SEQ],
        TEST_SEQ,
    )
)
THRESHOLDS = [10e-6, 5e-6, 2e-6]
MOD_RATE = 0.5


@pytest.mark.parametrize("testcase", MASS_SEQ_DICT.items())
@pytest.mark.parametrize("threshold", THRESHOLDS)
def test_testcase_with_recursion(testcase, threshold):
    dp_table = DynamicProgrammingTable(
        EXPLANATION_MASSES,
        reduced_table=True,
        reduced_set=False,
        compression_rate=32,
        tolerance=threshold,
        precision=TOLERANCE,
    )

    predicted_mass_explanations = explain_mass_with_recursion(
        testcase[0],
        dp_table=dp_table,
        seq_len=len(testcase[1]),
        max_modifications=round(MOD_RATE * len(testcase[1])),
    ).explanations

    assert predicted_mass_explanations is not None

    explanations = [tuple(expl) for expl in predicted_mass_explanations]

    assert tuple(testcase[1]) in explanations


WITH_MEMO = [True]
COMPRESSION_RATES = [32]


@pytest.mark.parametrize("testcase", MASS_SEQ_DICT.items())
@pytest.mark.parametrize("compression", COMPRESSION_RATES)
@pytest.mark.parametrize("memo", WITH_MEMO)
@pytest.mark.parametrize("threshold", THRESHOLDS)
def test_testcase_with_table(testcase, compression, threshold, memo):
    dp_table = DynamicProgrammingTable(
        EXPLANATION_MASSES,
        reduced_table=True,
        reduced_set=False,
        compression_rate=compression,
        tolerance=threshold,
        precision=TOLERANCE,
    )

    predicted_mass_explanations = explain_mass_with_table(
        testcase[0],
        dp_table=dp_table,
        seq_len=len(testcase[1]),
        max_modifications=round(MOD_RATE * len(testcase[1])),
        with_memo=memo,
    ).explanations

    assert predicted_mass_explanations is not None

    explanations = [tuple(expl) for expl in predicted_mass_explanations]

    assert tuple(testcase[1]) in explanations
