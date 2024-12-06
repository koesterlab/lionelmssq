from dataclasses import dataclass
from typing import Set, Tuple
import numpy as np
from lionelmssq.masses import MASSES
import polars as pl

from lionelmssq.masses import EXPLANATION_MASSES
from lionelmssq.masses import TOLERANCE

@dataclass
class MassExplanations:
    explanations: Set[Tuple[str]]

def explain_mass(mass: float) -> MassExplanations:
    
    '''
    Returns all the possible combinations of nucleosides that could sum up to the given mass.
    '''

    coins = EXPLANATION_MASSES.select(pl.col("coins")).to_numpy().flatten().tolist()

    #Convert the targets and coins to integers for easy opearations    
    target = int(round(mass/TOLERANCE,0))

    #Sort the coins, makes life easier
    coins.sort()
    
    # Memoization dictionary to store results for a given target
    memo = {}

    def dp(remaining, start):

        #TODO: Can check if there are any possible ways to achieve remaining using coins[start:], if NOT, don't execute the dict check, nor store the values (empty list) in the dictionary!

        # If the result for this state is already computed, return it
        if (remaining, start) in memo:
            return memo[(remaining, start)]

        # Base case: if target is zero, return a list with one empty combination
        if remaining == 0:
            return [[]]

        # Base case: if target is negative, no combinations possible
        if remaining < 0:
            return []

        # List to store all combinations for this state
        combinations = []

        # Try each coin starting from the current position to avoid duplicates
        for i in range(start, len(coins)):
            coin = coins[i]
            # Recurse with reduced target and the current coin
            sub_combinations = dp(remaining - coin, i)
            # Add current coin to all sub-combinations
            for combo in sub_combinations:
                combinations.append([coin] + combo)
        
        # Store result in memo
        memo[(remaining, start)] = combinations
        
        return combinations 

    # Start with the full target and all coins
    solution_coins = dp(target, 0)

    #Convert the coins to the respective nucleoside names
    solution_names  = set()

    for combo in solution_coins:
        
        #solution_names.add(tuple(EXPLANATION_MASSES.filter(pl.col("coins").is_in(combo)).select(pl.col("nucleoside")).to_numpy().flatten().tolist()))
        #TODO: Avoid this loop, somehow get column values directly from the polars dataframe!
        temp_list  = []
        for coin_val in combo:
            temp_list.append(EXPLANATION_MASSES.filter(pl.col("coins") == coin_val).select(pl.col("nucleoside")).item())

        solution_names.add(tuple(temp_list))
    
    #Return the desired Dataclass object
    return MassExplanations(explanations = solution_names)

# TODO: The case of 2A does not work with Tol of 1e-2, to be discussed with Johannes. Small rounding issue

#Test Cases!
#print(explain_mass(1563.52067))
#print(explain_mass(267.09675*2)) #This case doesn't properly work!
#print(explain_mass(283.09167*2))
