import pytest
from clr_loader import get_mono
from spectrseqtools.preprocessing.preprocessing import AveragineBackbone, set_averagine

rt = get_mono()

AVERAGINE = {
    AveragineBackbone.NONE: {
        "C": 9.5,
        "H": 12.75,
        "N": 3.75,
        "O": 5.0,
        "P": 0.0,
        "S": 0.0,
    },
    AveragineBackbone.PHOSPHATE: {
        "C": 9.5,
        "H": 12.75,
        "N": 3.75,
        "O": 7.0,
        "P": 1.0,
        "S": 0.0,
    },
    AveragineBackbone.THIOPHOSPHATE: {
        "C": 9.5,
        "H": 12.75,
        "N": 3.75,
        "O": 6.0,
        "P": 1.0,
        "S": 1.0,
    },
}


@pytest.mark.parametrize("backbone", AVERAGINE.keys())
def test_set_averagine(backbone):
    averagine = set_averagine(backbone=backbone)
    assert averagine == AVERAGINE[backbone]
