import pytest
from spectrseqtools.masses import PRECISION
from spectrseqtools.nucleotide_alphabet import NucleotideAlphabet
from spectrseqtools.prediction.composition_inference import (
    CompositionInferrer,
    SequenceInformation,
    infer_compositions_with_matrix,
    infer_compositions_with_recursion,
)

TEST_SEQ = [
    tuple("A"),
    ("A", "A"),
    ("G", "G"),
    ("C", "C"),
    ("U", "U"),
    ("C", "U", "A", "G"),
    ("C", "C", "U", "A", "G", "G"),
]
TOLERANCES = [10e-6, 5e-6, 2e-6]
MOD_RATE = 0.5


@pytest.mark.parametrize("seq", TEST_SEQ)
@pytest.mark.parametrize("tolerance", TOLERANCES)
def test_infer_composition_with_recursion(seq, tolerance):
    alphabet = NucleotideAlphabet.from_file()

    seq_weight = alphabet.get_seq_weight(seq)

    seq_info = SequenceInformation(
        max_len=int(seq_weight / alphabet.min_mass()),
        su_mass=seq_weight,
        obs_mass=seq_weight,
        modification_rate=MOD_RATE,
    )

    inferrer = CompositionInferrer(
        nucleotide_df=alphabet.nucleotides,
        compression_rate=32,
        tolerance=tolerance,
        precision=PRECISION,
        seq=seq_info,
    )

    predicted_compositions = infer_compositions_with_recursion(
        seq_weight,
        inferrer=inferrer,
        max_modifications=round(MOD_RATE * len(seq)),
    ).compositions

    assert predicted_compositions is not None

    compositions = [tuple(comp) for comp in predicted_compositions]

    assert tuple(seq) in compositions


WITH_MEMO = [True]
COMPRESSION_RATES = [32]


@pytest.mark.parametrize("seq", TEST_SEQ)
@pytest.mark.parametrize("compression", COMPRESSION_RATES)
@pytest.mark.parametrize("memo", WITH_MEMO)
@pytest.mark.parametrize("tolerance", TOLERANCES)
def test_infer_composition_with_matrix(seq, compression, tolerance, memo):
    alphabet = NucleotideAlphabet.from_file()

    seq_weight = alphabet.get_seq_weight(seq)

    seq_info = SequenceInformation(
        max_len=int(seq_weight / alphabet.min_mass()),
        su_mass=seq_weight,
        obs_mass=seq_weight,
        modification_rate=MOD_RATE,
    )

    inferrer = CompositionInferrer(
        nucleotide_df=alphabet.nucleotides,
        compression_rate=compression,
        tolerance=tolerance,
        precision=PRECISION,
        seq=seq_info,
    )

    predicted_compositions = infer_compositions_with_matrix(
        seq_weight,
        inferrer=inferrer,
        max_modifications=round(MOD_RATE * len(seq)),
        with_memo=memo,
    ).compositions

    assert predicted_compositions is not None

    compositions = [tuple(comp) for comp in predicted_compositions]

    assert tuple(seq) in compositions
