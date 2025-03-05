# import importlib.resources
# import os
import itertools
import pytest
# import yaml

from lionelmssq.mass_explanation import explain_mass, explain_mass_with_dp

TEST_MASSES_WITH_BACKBONE = [
    267.09675 + 62,          # A
    267.09675 * 2 + 62 * 2,  # AA
    283.09167 * 2 + 62 * 2,  # GG
    243.08552 * 2 + 62 * 2,  # CC
    244.06954 * 2 + 62 * 2,  # UU
    1037.34348 + 62 * 4,     # AUGC
    1563.52067 + 62 * 6,     # CCUAGG
]

TEST_MASSES = [
    267.09675,      # A
    267.09675 * 2,  # AA
    283.09167 * 2,  # GG
    243.08552 * 2,  # CC
    244.06954 * 2,  # UU
    1037.34348,     # AUGC
    1563.52067,     # CCUAGG
]

TEST_SEQ = [
    {"c-y": ("A")},
    {"c-y": ("A", "A")},
    {"c-y": ("G", "G")},
    {"c-y": ("C", "C")},
    {"c-y": ("U", "U")},
    {"c-y": ("A", "U", "G", "C")},
    {"c-y": ("A", "U", "G", "G", "C", "C")},
]

MASS_SEQ_DICT = dict(zip(TEST_MASSES_WITH_BACKBONE, TEST_SEQ))


# _TESTCASES = importlib.resources.files("tests") / "testcases"
# @pytest.mark.parametrize("testcase", _TESTCASES.iterdir())
@pytest.mark.parametrize("testcase", MASS_SEQ_DICT.items())
def test_testcase(testcase):
    # base_path = _TESTCASES / testcase
    # with open(base_path / "meta.yaml", "r") as f:
    #    meta = yaml.safe_load(f)

    predicted_mass_explanation = explain_mass(testcase[0])
    # print(predicted_mass_explanation)

    breakage = list(testcase[1].keys())[0]

    # Need to check if any possible permutation of testcase_permutations is in predicted_mass_explaination
    testcase_permutations = tuple(itertools.permutations(testcase[1][breakage]))

    assert any(
        perm in predicted_mass_explanation.explanations[breakage]
        for perm in testcase_permutations
    )

@pytest.mark.parametrize("testcase", MASS_SEQ_DICT.items())
def test_testcase_with_dp(testcase):
    # base_path = _TESTCASES / testcase
    # with open(base_path / "meta.yaml", "r") as f:
    #    meta = yaml.safe_load(f)

    predicted_mass_explanation = explain_mass_with_dp(testcase[0], True)
    # print(predicted_mass_explanation)

    breakage = list(testcase[1].keys())[0]

    # Need to check if any possible permutation of testcase_permutations is in predicted_mass_explaination
    testcase_permutations = tuple(itertools.permutations(testcase[1][breakage]))

    assert any(
        perm in predicted_mass_explanation.explanations[breakage]
        for perm in testcase_permutations
    )

@pytest.mark.parametrize("testcase", MASS_SEQ_DICT.items())
def test_testcase_with_dp_and_memo(testcase):
    # base_path = _TESTCASES / testcase
    # with open(base_path / "meta.yaml", "r") as f:
    #    meta = yaml.safe_load(f)

    predicted_mass_explanation = explain_mass_with_dp(testcase[0], False)
    # print(predicted_mass_explanation)

    breakage = list(testcase[1].keys())[0]

    # Need to check if any possible permutation of testcase_permutations is in predicted_mass_explaination
    testcase_permutations = tuple(itertools.permutations(testcase[1][breakage]))

    assert any(
        perm in predicted_mass_explanation.explanations[breakage]
        for perm in testcase_permutations
    )
