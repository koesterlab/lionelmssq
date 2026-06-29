import pytest
from spectrseqtools.compositions import Composition
from spectrseqtools.dataclasses import SequenceInformation
from spectrseqtools.error_calculator import ErrorCalculator
from spectrseqtools.nucleotide_alphabet import NucleotideAlphabet
from spectrseqtools.prediction.composition_inference import (
    CompositionInferrer,
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
WITH_MEMO = [True]
COMPRESSION_RATES = [32]


@pytest.mark.parametrize("seq", TEST_SEQ)
@pytest.mark.parametrize("tolerance", TOLERANCES)
def test_infer_composition_with_recursion(seq, tolerance):
    error_calculator = ErrorCalculator.with_metric(tolerance=tolerance)
    alphabet = NucleotideAlphabet.from_file(
        modification_rate=MOD_RATE, error=error_calculator
    )
    seq_weight = alphabet.get_seq_weight(seq)

    seq_info = SequenceInformation(
        max_len=int(seq_weight / alphabet.min.nucleotide_mass),
        su_mass=seq_weight,
        obs_mass=seq_weight,
        modification_rate=MOD_RATE,
    )

    inferrer = CompositionInferrer(
        alphabet=alphabet,
        compression_rate=32,
        error=error_calculator,
        seq=seq_info,
    )

    compositions = infer_compositions_with_recursion(mass=seq_weight, inferrer=inferrer)

    assert len(compositions) != 0
    assert Composition(*tuple(alphabet.get_idx(nuc) for nuc in seq)) in compositions


@pytest.mark.parametrize("seq", TEST_SEQ)
@pytest.mark.parametrize("compression", COMPRESSION_RATES)
@pytest.mark.parametrize("memo", WITH_MEMO)
@pytest.mark.parametrize("tolerance", TOLERANCES)
def test_infer_composition_with_matrix(seq, compression, tolerance, memo):
    error_calculator = ErrorCalculator.with_metric(tolerance=tolerance)
    alphabet = NucleotideAlphabet.from_file(
        modification_rate=MOD_RATE, error=error_calculator
    )
    seq_weight = alphabet.get_seq_weight(seq)

    seq_info = SequenceInformation(
        max_len=int(seq_weight / alphabet.min.nucleotide_mass),
        su_mass=seq_weight,
        obs_mass=seq_weight,
        modification_rate=MOD_RATE,
    )

    inferrer = CompositionInferrer(
        alphabet=alphabet,
        compression_rate=compression,
        error=error_calculator,
        seq=seq_info,
    )

    compositions = inferrer.infer_compositions(mass=seq_weight, with_memo=memo)

    assert len(compositions) != 0
    assert Composition(*tuple(alphabet.get_idx(nuc) for nuc in seq)) in compositions
