from dataclasses import dataclass
from typing import List, Set

from lionelmssq.masses import MASSES


@dataclass
class MassExplanations:
    explanations: Set[List[str]]

def explain_mass(mass: float) -> MassExplanations:
    
    ...