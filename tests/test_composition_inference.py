import pytest
import polars as pl

from spectrseqtools.masses import (
    NUCLEOTIDE_DF,
    PHOSPHATE_LINK_MASS,
    PRECISION,
)
from spectrseqtools.prediction.composition_inference import (
    infer_compositions_with_recursion,
    infer_compositions_with_matrix,
)
from spectrseqtools.prediction.traceback_matrix import (
    CompositionInferrer,
    SequenceInformation,
)


def get_seq_weight(seq: tuple) -> float:
    seq_df = pl.DataFrame(data=seq, schema=["name"])
    seq_df = seq_df.with_columns(
        pl.col("name")
        .map_elements(
            lambda x: (
                NUCLEOTIDE_DF.filter(pl.col("representative") == x)
                .get_column("nucleoside_mass")
                .to_list()[0]
            ),
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
TOLERANCES = [10e-6, 5e-6, 2e-6]
MOD_RATE = 0.5


@pytest.mark.parametrize("testcase", MASS_SEQ_DICT.items())
@pytest.mark.parametrize("tolerance", TOLERANCES)
def test_infer_composition_with_recursion(testcase, tolerance):
    seq_info = SequenceInformation(
        max_len=int(
            testcase[0]
            / PRECISION
            / min(pl.Series(NUCLEOTIDE_DF.select("integer_mass")).to_list())
        ),
        su_mass=testcase[0],
        obs_mass=testcase[0],
        modification_rate=MOD_RATE,
    )

    inferrer = CompositionInferrer(
        nucleotide_df=NUCLEOTIDE_DF,
        compression_rate=32,
        tolerance=tolerance,
        precision=PRECISION,
        seq=seq_info,
    )

    predicted_compositions = infer_compositions_with_recursion(
        testcase[0],
        inferrer=inferrer,
        max_modifications=round(MOD_RATE * len(testcase[1])),
    ).compositions

    assert predicted_compositions is not None

    compositions = [tuple(comp) for comp in predicted_compositions]

    assert tuple(testcase[1]) in compositions


WITH_MEMO = [True]
COMPRESSION_RATES = [32]


@pytest.mark.parametrize("testcase", MASS_SEQ_DICT.items())
@pytest.mark.parametrize("compression", COMPRESSION_RATES)
@pytest.mark.parametrize("memo", WITH_MEMO)
@pytest.mark.parametrize("tolerance", TOLERANCES)
def test_infer_composition_with_matrix(testcase, compression, tolerance, memo):
    seq_info = SequenceInformation(
        max_len=int(
            testcase[0]
            / PRECISION
            / min(pl.Series(NUCLEOTIDE_DF.select("integer_mass")).to_list())
        ),
        su_mass=testcase[0],
        obs_mass=testcase[0],
        modification_rate=MOD_RATE,
    )

    inferrer = CompositionInferrer(
        nucleotide_df=NUCLEOTIDE_DF,
        compression_rate=compression,
        tolerance=tolerance,
        precision=PRECISION,
        seq=seq_info,
    )

    predicted_compositions = infer_compositions_with_matrix(
        testcase[0],
        inferrer=inferrer,
        max_modifications=round(MOD_RATE * len(testcase[1])),
        with_memo=memo,
    ).compositions

    assert predicted_compositions is not None

    compositions = [tuple(comp) for comp in predicted_compositions]

    assert tuple(testcase[1]) in compositions
