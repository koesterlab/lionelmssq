# import importlib.resources
# import os
import itertools
import pytest
#import yaml

from lionelmssq.mass_explanation import explain_mass

TEST_MASSES = [
    267.09675 * 2,  # 2A
    283.09167 * 2,  # 2G
    243.08552 * 2,  # 2C
    244.06954 * 2,  # 2U
    1037.34348,  # AUGC
    1563.52067,  # CCUAGG
]

TEST_SEQ = [
    ("A", "A"),
    ("G", "G"),
    ("C", "C"),
    ("U", "U"),
    ("A", "U", "G", "C"),
    ("A", "U", "G", "G", "C", "C"),
]

MASS_SEQ_DICT = dict(zip(TEST_MASSES, TEST_SEQ))


# _TESTCASES = importlib.resources.files("tests") / "testcases"
# @pytest.mark.parametrize("testcase", _TESTCASES.iterdir())
@pytest.mark.parametrize("testcase", MASS_SEQ_DICT.items())
def test_testcase(testcase):
    # base_path = _TESTCASES / testcase
    # with open(base_path / "meta.yaml", "r") as f:
    #    meta = yaml.safe_load(f)

    predicted_mass_explaination = explain_mass(testcase[0])

    # Need to check if any possible permutation of testcase_permutations is in predicted_mass_explaination
    testcase_permutations = tuple(itertools.permutations(testcase[1]))

    assert any(
        perm in predicted_mass_explaination.explanations
        for perm in testcase_permutations
    )
