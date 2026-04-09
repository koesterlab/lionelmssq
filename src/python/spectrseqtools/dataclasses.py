# -*- coding: utf-8 -*-
"""Module with dataclasses."""

from dataclasses import dataclass


@dataclass
class SolverParameters:
    """Class for parameters used to solve optimization problems."""

    solver: str
    threads: int
    msg: bool
    time_limit_short: int
    time_limit_long: int

    def to_dict(self, filter_only: bool = False) -> dict:
        """Return dictionary of solver parameters.

        Parameters
        ----------
        filter_only : bool
            Flag whether solver is only used as a filter (not prediction).

        Returns
        -------
        dict
            Dictionary containing solver parameters.

        """
        # Retrieve parameters from class
        params = self.__dict__.copy()

        # Set time limit based on flag
        if filter_only:
            params["timeLimit"] = params.pop("time_limit_short")
            params.pop("time_limit_long")
        else:
            params["timeLimit"] = params.pop("time_limit_long")
            params.pop("time_limit_short")

        return params
