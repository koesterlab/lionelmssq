# -*- coding: utf-8 -*-
"""Module with enum classes."""

from enum import Enum


class AveragineBackbone(Enum):
    """Enum of backbone types for Averagine model used in deisotoping."""

    NONE = "no_backbone"
    PHOSPHATE = "phosphate"
    THIOPHOSPHATE = "thiophosphate"


class ErrorMetric(Enum):
    """Enum of types of metrics used for error calculations over multiple values."""

    L1NORM = "l1_norm"
    L2NORM = "l2_norm"


class SolverType(Enum):
    """Enum of types of solvers used for linear optimization."""

    CBC = "cbc"
    GUROBI = "gurobi"
    HIGHS = "highs"
