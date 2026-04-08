# -*- coding: utf-8 -*-
"""Module with enum classes."""

from enum import Enum


class AveragineBackbone(Enum):
    """Enum of backbone types for Averagine model used in deisotoping."""

    NONE = "no_backbone"
    PHOSPHATE = "phosphate"
    THIOPHOSPHATE = "thiophosphate"


class SolverType(Enum):
    """Enum of types of solvers used for linear optimization."""

    CBC = "cbc"
    GUROBI = "gurobi"
