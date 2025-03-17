# import importlib.resources
# import os
import itertools
import pytest
# import yaml

from lionelmssq.mass_explanation import explain_mass, explain_mass_with_dp

TEST_MASSES_WITH_BACKBONE = [
    267.09675 + 61.95577,          # A
    267.09675 * 2 + 61.95577 * 2,  # AA
    283.09167 * 2 + 61.95577 * 2,  # GG
    243.08552 * 2 + 61.95577 * 2,  # CC
    244.06954 * 2 + 61.95577 * 2,  # UU
    1037.34348 + 61.95577 * 4,     # AUGC
    1563.52067 + 61.95577 * 6,     # CCUAGG
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
    {"c/y_c/y": ("A")},
    {"c/y_c/y": ("A", "A")},
    {"c/y_c/y": ("G", "G")},
    {"c/y_c/y": ("C", "C")},
    {"c/y_c/y": ("U", "U")},
    {"c/y_c/y": ("C", "U", "A", "G")},
    {"c/y_c/y": ("C", "C", "U", "A", "G", "G")},
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

    breakage = list(testcase[1].keys())[0]
    sorted_explanations = [tuple(solution) for solution in
                           predicted_mass_explanation.explanations[breakage]]

    assert (tuple(testcase[1][breakage]) in sorted_explanations)

@pytest.mark.parametrize("testcase", MASS_SEQ_DICT.items())
def test_testcase_with_dp(testcase):
    predicted_mass_explanation = explain_mass_with_dp(testcase[0], False)

    breakage = list(testcase[1].keys())[0]
    sorted_explanations = [tuple(solution) for solution in
                           predicted_mass_explanation.explanations[breakage]]

    assert (tuple(testcase[1][breakage]) in sorted_explanations)

@pytest.mark.parametrize("testcase", MASS_SEQ_DICT.items())
def test_testcase_with_dp_and_memo(testcase):
    predicted_mass_explanation = explain_mass_with_dp(testcase[0], True)

    breakage = list(testcase[1].keys())[0]
    sorted_explanations = [tuple(solution) for solution in
                           predicted_mass_explanation.explanations[breakage]]

    assert (tuple(testcase[1][breakage]) in sorted_explanations)
